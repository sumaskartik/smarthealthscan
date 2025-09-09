# main.py
import shutil, tempfile, os
from fastapi import FastAPI, File, UploadFile
from rag import extract_from_image_with_llm
from db import master_collection


app = FastAPI(debug=True)


def get_document_schema(doc_type: str):
    """
    Retrieves the UI schema for a given document type from the database.
    """
    try:
        schema_doc = master_collection.find_one(
            {"doc_type": doc_type}, {"_id": 0, "fields": 1}
        )
        if schema_doc and "fields" in schema_doc:
            return schema_doc["fields"]
        return None
    except Exception as e:
        return None


@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart Health Scan API. Use /add_document to upload documents or /test to test the API."
    }


@app.post("/extract-and-validate")
async def add_document(file: UploadFile = File(...), doc_type: str = "KYC_document"):
    ext = os.path.splitext(file.filename)[1] or ".bin"

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    schema_fields = get_document_schema(doc_type)

    result = extract_from_image_with_llm(tmp_path, schema_fields)

    return {"message": result}


@app.get("/test")
async def chat(doc_type: str = "lab_report"):
    # Fetch documents with doc_type="health" and exclude _id
    docs = list(
        master_collection.find(
            {"doc_type": doc_type},  # query filter
            {"_id": 0, "fields": 1},  # projection
        )
    )

    return docs
