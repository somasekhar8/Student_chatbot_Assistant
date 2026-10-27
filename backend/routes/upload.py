"""POST /upload - index a PDF/TXT for one session."""
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.rag.document_loader import MAX_UPLOAD_MB, load_document
from src.rag.text_splitter import split_documents
from src.rag.vector_store import get_index_store
from src.schemas.chat_schema import SESSION_ID_PATTERN, DocumentsResponse, UploadResponse

router = APIRouter(tags=["documents"])


def _check_session(session_id: str) -> str:
    if not re.match(SESSION_ID_PATTERN, session_id or ""):
        raise HTTPException(status_code=422, detail="session_id must be 8-64 letters, digits, '-' or '_'.")
    return session_id


@router.post("/upload", response_model=UploadResponse)
def upload_document(session_id: str = Form(...), file: UploadFile = File(...)) -> UploadResponse:
    """Read the file, split it into chunks, embed them and store them in this session's FAISS index."""
    session_id = _check_session(session_id)
    data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)  # never read more than the limit

    pages = load_document(data, file.filename or "document")  # raises DocumentError -> HTTP 400
    name = pages[0].metadata["source"]
    chunks = split_documents(pages)
    added = get_index_store().add_documents(session_id, chunks, name)

    return UploadResponse(
        session_id=session_id,
        filename=name,
        pages=len(pages),
        chunks=len(chunks),
        already_indexed=not added,
        documents=get_index_store().list_files(session_id),
    )


@router.get("/documents/{session_id}", response_model=DocumentsResponse)
def list_documents(session_id: str) -> DocumentsResponse:
    session_id = _check_session(session_id)
    return DocumentsResponse(session_id=session_id, documents=get_index_store().list_files(session_id))
