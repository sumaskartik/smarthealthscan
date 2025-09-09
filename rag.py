import base64
import json
import re
import logging
import os
import dotenv
import mimetypes
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load .env file
dotenv.load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in your .env file.")

# Create vision client with key
VISION_CLIENT = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


def extract_from_image_with_llm(
    file_path: str, schema_fields: List[Dict]
) -> Dict[str, Any]:
    """Send image directly to vision model and extract fields."""

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
        if mime_type not in ["image/png", "image/jpeg", "image/gif", "image/webp"]:
            raise ValueError(
                f"❌ Unsupported format: {mime_type}. Convert before sending."
            )

        # Read and base64 encode image
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Send to Vision LLM
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

        # Try extracting JSON from response
        json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = {}

        # Ensure all schema fields are present, missing ones get None
        for field in field_names:
            parsed.setdefault(field, None)

        return parsed

    except json.JSONDecodeError:
        logging.error(f"❌ Failed to decode JSON from LLM response: {raw_content}")
        return {f: None for f in field_names}

    except Exception as e:
        logging.error(f"❌ Error during extraction: {e}")
        return {f: None for f in field_names}
