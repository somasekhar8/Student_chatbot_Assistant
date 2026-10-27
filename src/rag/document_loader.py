"""Load PDF / TXT uploads into LangChain Documents with file name and page metadata.

Reused from the original project: PyMuPDFLoader for PDFs and TextLoader for TXT.
New: works on raw bytes (so FastAPI and Streamlit can both use it), checks the
file, and stores a 1-based `page` number so citations read "page 3" not "page 2".
"""
import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = (".pdf", ".txt")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))


class DocumentError(ValueError):
    """Raised for bad uploads. The message is safe to show to students."""


def clean_filename(filename: str) -> str:
    """Keep only the base name so paths like ../../x.pdf can never be used."""
    name = os.path.basename((filename or "").replace("\\", "/")).replace("\x00", "").strip()
    return name[:120] or "document"


def load_document(data: bytes, filename: str) -> List[Document]:
    """Turn uploaded bytes into one Document per page (PDF) or one for a TXT file."""
    name = clean_filename(filename)
    extension = os.path.splitext(name)[1].lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"'{name}' is not supported. Please upload a PDF or TXT file.")
    if not data:
        raise DocumentError(f"'{name}' is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(f"'{name}' is larger than {MAX_UPLOAD_MB} MB.")
    if extension == ".pdf" and not data.lstrip()[:5] == b"%PDF-":
        raise DocumentError(f"'{name}' does not look like a valid PDF file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(data)
        temp_path = temp_file.name

    try:
        if extension == ".pdf":
            # PyMuPDF is much faster than PyPDF.
            raw_documents = PyMuPDFLoader(temp_path).load()
        else:
            try:
                raw_documents = TextLoader(temp_path, encoding="utf-8").load()
            except UnicodeDecodeError:
                raw_documents = TextLoader(temp_path, encoding="latin-1").load()
    except Exception as exc:
        raise DocumentError(f"Could not read '{name}'. The file may be damaged or password protected.") from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    documents: List[Document] = []
    for index, doc in enumerate(raw_documents):
        text = (doc.page_content or "").strip()
        if not text:
            continue  # skip blank pages
        zero_based_page = doc.metadata.get("page", index)
        documents.append(
            Document(
                page_content=text,
                metadata={"source": name, "page": int(zero_based_page) + 1 if extension == ".pdf" else 1},
            )
        )

    if not documents:
        raise DocumentError(
            f"No readable text was found in '{name}'. Scanned (image-only) PDFs are not supported."
        )
    return documents
