import os
import dotenv
import json
import re
from PIL import Image
import pytesseract
import cv2
import numpy as np
from deskew import determine_skew
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from masteruijson import MASTER_UI_SCHEMAS
from db import master_collection

# --------------------------
# Load .env
# --------------------------
dotenv.load_dotenv()


def preprocess_image(image_path: str):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image at {image_path} not found.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = determine_skew(gray)
        if angle is not None:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(thresh)
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None


def clean_ocr_text(text: str) -> str:
    text = text.replace("'Q'", "O").replace("’O’", "O").replace("Q", "O")
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------------------------------------------
# File loader
# -------------------------------------------------------------------
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
    elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
        pil_image = preprocess_image(file_path)
        if pil_image:
            text = pytesseract.image_to_string(pil_image, config="--oem 3 --psm 6")
            text = clean_ocr_text(text)
            metadata = {"source": os.path.basename(file_path)}
            docs = [Document(page_content=text, metadata=metadata)]
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")

    return docs


# -------------------------------------------------------------------
# LLM helpers
# -------------------------------------------------------------------
def detect_doc_type(context: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = f"""
You are a document classifier. Decide the document type.
Options: ["bill", "prescription", "lab_report", "id_proof"]

Document text:
{context}

Answer with only the type from the options.
"""
    resp = llm.invoke(prompt)
    return resp.content.strip().lower()


def extract_with_llm(docs, doc_type: str, schema_fields):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    context = "\n".join([doc.page_content for doc in docs])
    schema_field_names = [f['fieldName'] for f in schema_fields]

    prompt = f"""
You are given a document of type {doc_type}.
Extract the values for the following fields: {schema_field_names}.
Return strictly valid JSON where the keys are the field names and values are extracted.

Document text:
{context}
"""

    response = llm.invoke(prompt)
    raw = response.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}

    return data

def get_health_documents(doc_type: str):
    docs = list(
            master_collection.find(
                {"doc_type": doc_type},  # query filter
                {"_id": 0, "fields": 1},  # projection
            )
        )
    return docs
# -------------------------------------------------------------------
# Unified pipeline
# -------------------------------------------------------------------
def process_document_and_extract(file_path: str):
    try:
        docs = load_document(file_path)
        if not docs:
            return {"error": f"⚠️ Could not extract any text from {os.path.basename(file_path)}."}

        context = "".join(doc.page_content for doc in docs)
        if not context.strip():
            return {"error": "⚠️ Document is empty or contains no readable text."}

        # Step 1: Detect doc type
        doc_type = detect_doc_type(context)
        if doc_type not in MASTER_UI_SCHEMAS:
            return {"error": f"Unsupported or unknown doc type detected: {doc_type}"}

        # schema_fields = MASTER_UI_SCHEMAS[doc_type]
        schema_fields = get_health_documents(doc_type)[0]['fields']
        # Step 2: Extract data
        extracted_data = extract_with_llm(docs, doc_type, schema_fields)

        # Step 3: Map into schema
        extracted, missing = [], []

        for field in schema_fields:
            field_copy = dict(field)  # shallow copy
            value = extracted_data.get(field["fieldName"])
            if value is not None and str(value).strip() != "":
                field_copy["value"] = [value]
                extracted.append(field_copy)
            else:
                field_copy["value"] = []
                missing.append(field_copy)

        return {"extracted": extracted, "missing": missing}

    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}
