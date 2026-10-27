"""Phase 1 tests: loading, splitting, per-session indexes, relevance filtering."""
import pytest

from src.rag.document_loader import DocumentError, load_document
from src.rag.retriever import retrieve
from src.rag.text_splitter import split_documents
from src.rag.vector_store import get_index_store


def index(session_id, data, name):
    pages = load_document(data, name)
    chunks = split_documents(pages)
    return get_index_store().add_documents(session_id, chunks, name)


def test_pdf_pages_are_numbered_from_one(sample_pdf):
    pages = load_document(sample_pdf, "ml.pdf")
    assert [p.metadata["page"] for p in pages] == [1, 2]
    assert all(p.metadata["source"] == "ml.pdf" for p in pages)


def test_txt_file_is_supported():
    pages = load_document(b"Plain text notes about databases.", "notes.txt")
    assert pages[0].metadata == {"source": "notes.txt", "page": 1}


def test_path_in_filename_is_removed():
    pages = load_document(b"hello world notes", "../../etc/secret.txt")
    assert pages[0].metadata["source"] == "secret.txt"


@pytest.mark.parametrize(
    "data,name",
    [
        (b"", "empty.txt"),
        (b"not a pdf", "fake.pdf"),
        (b"data", "program.exe"),
    ],
)
def test_bad_uploads_raise_friendly_errors(data, name):
    with pytest.raises(DocumentError):
        load_document(data, name)


def test_chunks_keep_page_metadata(sample_pdf):
    chunks = split_documents(load_document(sample_pdf, "ml.pdf"), chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 2
    assert {c.metadata["page"] for c in chunks} == {1, 2}


def test_retrieval_returns_source_and_page(sample_pdf):
    index("session-aaaa", sample_pdf, "ml.pdf")
    result = retrieve("session-aaaa", "What are the types of machine learning?")
    assert result.found_relevant
    top = result.chunks[0]
    assert top.source == "ml.pdf"
    assert top.page == 2
    assert top.score >= 0.30


def test_irrelevant_question_returns_no_chunks(sample_pdf):
    index("session-aaaa", sample_pdf, "ml.pdf")
    result = retrieve("session-aaaa", "Who won the football world cup?")
    assert result.has_documents
    assert not result.found_relevant


def test_sessions_are_isolated(sample_pdf):
    index("session-aaaa", sample_pdf, "ml.pdf")
    # Another session has no documents, so nothing can leak.
    other = retrieve("session-bbbb", "machine learning")
    assert not other.has_documents
    assert other.chunks == []
    assert get_index_store().list_files("session-bbbb") == []


def test_same_file_is_not_indexed_twice(sample_pdf):
    assert index("session-aaaa", sample_pdf, "ml.pdf") is True
    assert index("session-aaaa", sample_pdf, "ml.pdf") is False
    assert get_index_store().list_files("session-aaaa") == ["ml.pdf"]


def test_clear_removes_documents(sample_pdf):
    index("session-aaaa", sample_pdf, "ml.pdf")
    get_index_store().clear("session-aaaa")
    assert not get_index_store().has_documents("session-aaaa")


def test_old_sessions_are_evicted():
    from src.rag.vector_store import SessionIndexStore
    from langchain_core.documents import Document

    small = SessionIndexStore(max_sessions=2)
    for name in ("one-session", "two-session", "three-session"):
        small.add_documents(name, [Document(page_content=f"text {name}", metadata={"source": "a.txt", "page": 1})], "a.txt")
    assert not small.has_documents("one-session")
    assert small.has_documents("three-session")
