from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .rag import OllamaClient, OllamaError, RAGService


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Local Document RAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
rag = RAGService(OllamaClient())


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "ollama": await rag.client.health(), "documents": len(rag.documents)}


@app.get("/api/documents")
async def list_documents() -> dict:
    return {"documents": [document.summary() for document in rag.documents.values()]}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".docx"}:
        raise HTTPException(status_code=415, detail="Please upload a PDF or DOCX file.")
    try:
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    finally:
        await file.close()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Files must be smaller than {settings.max_upload_mb} MB.")
    try:
        document = await rag.add_document(filename, data)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"document": document.summary()}


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str) -> dict:
    if not rag.remove_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": True}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    try:
        answer, sources = await rag.answer(request.question, request.top_k)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"answer": answer, "sources": sources}


@app.post("/api/retrieve")
async def retrieve(request: ChatRequest) -> dict:
    try:
        sources = await rag.retrieve(request.question, request.top_k)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"sources": sources}
