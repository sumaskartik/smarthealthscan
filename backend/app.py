# main.py
import shutil, tempfile, os
from fastapi import FastAPI, File, Form, UploadFile
from rag import  process_document_and_extract

app = FastAPI()

@app.get("/test")
async def test():
    return {"server": "running"}

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

# @app.post("/chat")
# async def chat(query: str = Form(...)):
#     result = run_rag(query)
#     return {"analysis": result}
