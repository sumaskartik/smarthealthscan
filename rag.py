import os
import json
import re
import logging
import dotenv
from typing import List, Dict, Any, Optional

# Third-party imports
import cv2
import numpy as np
import pytesseract
from PIL import Image
from deskew import determine_skew

# LangChain and Database imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from db import master_collection  # Assuming this is your MongoDB collection instance

# --------------------------
# Configuration & Constants
# --------------------------
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# LLM Configuration
LLM_MODEL = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0
CLASSIFICATION_MAX_CHARS = (
    4000  # Use first 4000 chars for classification to save tokens
)

# File Type Constants
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".docx", ".doc"}

# OCR Configuration
TESSERACT_CONFIG = "--oem 3 --psm 6"

# Initialize LLM client once to be reused
try:
    LLM_CLIENT = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
except Exception as e:
    logging.error(f"Failed to initialize ChatOpenAI client: {e}")
    LLM_CLIENT = None


# --------------------------
# Image Processing & OCR
# --------------------------
def preprocess_image(image_path: str) -> Optional[Image.Image]:
    """
    Loads, deskews, and applies preprocessing to an image for better OCR results.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Image not found or could not be read at path: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Deskew the image
        angle = determine_skew(gray)
        if angle is not None:
            (h, w) = gray.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(
                gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )

        # Apply thresholding for cleaner text
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(thresh)
    except Exception as e:
        logging.error(f"Error during image preprocessing for {image_path}: {e}")
        return None


def clean_ocr_text(text: str) -> str:
    """Common text cleaning for OCR output."""
    # Replace common OCR errors (e.g., 'Q' instead of 'O')
    text = text.replace("'Q'", "O").replace("’O’", "O").replace("Q", "O")
    # Normalize whitespace
    return re.sub(r"\s+", " ", text).strip()


# -------------------------------------------------------------------
# Document Loaders (Refactored for Modularity)
# -------------------------------------------------------------------
def _load_image(file_path: str) -> List[Document]:
    """Loads text from an image file using OCR."""
    pil_image = preprocess_image(file_path)
    if not pil_image:
        return []

    text = pytesseract.image_to_string(pil_image, config=TESSERACT_CONFIG)
    cleaned_text = clean_ocr_text(text)
    metadata = {"source": os.path.basename(file_path)}
    return [Document(page_content=cleaned_text, metadata=metadata)]


def _load_pdf(file_path: str) -> List[Document]:
    """Loads text from a PDF file."""
    return PyPDFLoader(file_path).load()


def _load_docx(file_path: str) -> List[Document]:
    """Loads text from a DOCX file."""
    return Docx2txtLoader(file_path).load()


LOADER_MAPPING = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,
    **{ext: _load_image for ext in SUPPORTED_IMAGE_EXTENSIONS},
}


def load_document(file_path: str) -> List[Document]:
    """
    Loads a document from a file path using the appropriate loader.
    """
    ext = os.path.splitext(file_path)[1].lower()
    loader = LOADER_MAPPING.get(ext)

    if not loader:
        raise ValueError(f"❌ Unsupported file type: {ext}")

    return loader(file_path)


# -------------------------------------------------------------------
# LLM Helper Functions
# -------------------------------------------------------------------
def detect_doc_type(context: str) -> str:
    """
    Detects the document type using a small chunk of the document context.
    """
    if not LLM_CLIENT:
        raise ConnectionError("LLM Client is not initialized.")

    # Use only the beginning of the context for efficient classification
    context_snippet = context[:CLASSIFICATION_MAX_CHARS]

    prompt = f"""
You are a document classifier. Your task is to identify the type of the document based on its content.
Choose from the following options: ["inpatient_bill", "diagnostic_report", "KYC_document", "lab_report"]

Document text snippet:
---
{context_snippet}
---

Based on the text, what is the document type? Answer with only the type from the options provided.
"""
    response = LLM_CLIENT.invoke(prompt)
    return response.content.strip().lower()


def extract_with_llm(
    context: str, doc_type: str, schema_fields: List[Dict]
) -> Dict[str, Any]:
    """
    Extracts structured data from the document context based on a given schema.
    """
    if not LLM_CLIENT:
        raise ConnectionError("LLM Client is not initialized.")

    field_names = [f["fieldName"] for f in schema_fields]

    prompt = f"""
You are an expert data extraction AI. You are given a document of type '{doc_type}'.
Your task is to extract the values for the following fields: {field_names}.
Return the result as a single, strictly valid JSON object. The keys of the JSON must be the field names, and the values should be the data you extracted.
If a value is not found for a field, omit it from the JSON.

Document text:
---
{context}
---

JSON Output:
"""
    response = LLM_CLIENT.invoke(prompt)
    raw_content = response.content.strip()

    try:
        # LLMs often wrap JSON in markdown backticks
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}  # Return empty dict if no JSON is found
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM response: {raw_content}")
        return {}  # Return empty on failure to prevent crash


def get_document_schema(doc_type: str) -> Optional[List[Dict]]:
    """
    Retrieves the UI schema for a given document type from the database.
    """
    try:
        schema_doc = master_collection.find_one(
            {"doc_type": doc_type}, {"_id": 0, "fields": 1}
        )
        if schema_doc and "fields" in schema_doc:
            return schema_doc["fields"]
        logging.warning(f"No schema found for document type: {doc_type}")
        return None
    except Exception as e:
        logging.error(f"Database error while fetching schema for {doc_type}: {e}")
        return None


# -------------------------------------------------------------------
# Main Processing Pipeline
# -------------------------------------------------------------------
def process_document_and_extract(file_path: str) -> Dict[str, Any]:
    """
    The main pipeline to load, classify, and extract data from a document.
    """
    try:
        # 1. Load document text
        docs = load_document(file_path)
        if not docs:
            return {
                "error": f"⚠️ Could not load or extract text from {os.path.basename(file_path)}."
            }

        context = "\n".join(doc.page_content for doc in docs)
        if not context.strip():
            return {"error": "⚠️ Document is empty or contains no readable text."}

        # 2. Detect document type
        doc_type = detect_doc_type(context)
        logging.info(f"Detected document type: {doc_type}")

        # 3. Retrieve schema from DB
        schema_fields = get_document_schema(doc_type)
        if not schema_fields:
            return {
                "error": f"Schema not found for document type '{doc_type}'. Cannot proceed."
            }

        # 4. Extract data using LLM
        extracted_data = extract_with_llm(context, doc_type, schema_fields)

        # 5. Map extracted data into the final schema structure
        extracted, missing = [], []
        for field in schema_fields:
            field_copy = field.copy()  # Use a copy to avoid modifying original schema
            value = extracted_data.get(field["fieldName"])

            # Check for a non-empty value
            if value is not None and str(value).strip() != "":
                field_copy["value"] = [value]  # Assuming value is a list
                extracted.append(field_copy)
            else:
                field_copy["value"] = []
                missing.append(field_copy)

        return {"extracted": extracted, "missing": missing}

    except ValueError as ve:
        logging.error(f"Validation Error: {ve}")
        return {"error": str(ve)}
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in the pipeline: {e}", exc_info=True
        )
        return {"error": f"An unexpected error occurred: {e}"}
