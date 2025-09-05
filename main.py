# main.py
import shutil, tempfile, os
from fastapi import FastAPI, File, UploadFile
from rag import process_document_and_extract
from db import master_collection

app = FastAPI(debug=True)


@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart Health Scan API. Use /add_document to upload documents or /test to test the API."
    }


@app.post("/add_document")
async def add_document(file: UploadFile = File(...)):
    # preserve file extension (pdf, docx, png, etc.)
    ext = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Add document to vector store
    result = process_document_and_extract(tmp_path)
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
