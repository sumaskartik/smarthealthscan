import base64
import json
import re
import logging
import os
import dotenv
import mimetypes
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

dotenv.load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env")

# Gemini vision model
VISION_CLIENT = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", temperature=0, api_key=api_key
)


def extract_from_image_with_gemini(
    file_path: str, schema_fields: List[Dict]
) -> Dict[str, Any]:
    """Send image to Gemini Vision model and extract fields."""

    field_names = [f["fieldName"] for f in schema_fields]

    prompt = f"""
You are an expert data extraction AI.
Extract values for these fields: {field_names}.
Return only valid JSON.
If value not found, set it to null.
For invoice/lab report items, return array of objects.
"""

    try:
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type not in ["image/png", "image/jpeg", "image/webp"]:
            raise ValueError(f"❌ Unsupported format: {mime_type}")

        # Base64 encode image
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Call Gemini
        response = VISION_CLIENT.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{img_b64}"},
                        },
                    ]
                )
            ]
        )

        raw_content = response.content.strip()

        # Try extracting JSON
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = {}

        # Ensure schema fields exist
        for field in field_names:
            parsed.setdefault(field, None)

        extracted, missing = [], []
        for field in schema_fields:
            field_copy = field.copy()  # Use a copy to avoid modifying original schema
            value = parsed.get(field["fieldName"])

            # Check for a non-empty value
            if value is not None and str(value).strip() != "":
                field_copy["value"] = [value]  # Assuming value is a list
                extracted.append(field_copy)
            else:
                field_copy["value"] = []
                missing.append(field_copy)

        return {"extracted": extracted, "missing": missing}

    except json.JSONDecodeError:
        logging.error(f"❌ Failed to decode JSON from Gemini response: {raw_content}")
        return {f: None for f in field_names}
    except Exception as e:
        logging.error(f"❌ Error during extraction: {e}")
        return {f: None for f in field_names}
