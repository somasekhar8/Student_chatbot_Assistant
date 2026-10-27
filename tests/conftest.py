import io
import sys
from pathlib import Path

import pytest

# Make `import src...` work no matter where pytest is started.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.fakes import FakeChat, HashingEmbeddings  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_app(monkeypatch):
    """Every test gets: a fresh document store, fresh chat memory and fake embeddings."""
    from src.memory import chat_history
    from src.rag import vector_store

    monkeypatch.setattr(vector_store, "get_embedding_model", lambda: HashingEmbeddings())
    monkeypatch.setattr(vector_store, "_store", vector_store.SessionIndexStore())
    chat_history.reset_checkpointer()
    for name in ("GROQ_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "PRIMARY_LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RELEVANCE_THRESHOLD", "0.30")
    yield


@pytest.fixture
def install_llm(monkeypatch):
    """Install a scripted fake model. Usage: llm = install_llm(lambda messages: "hello")."""

    def _install(reply_fn):
        model = FakeChat(reply_fn=reply_fn)
        monkeypatch.setattr("src.llm.fallback.available_models", lambda temperature=0.2: [model])
        return model

    return _install


def make_pdf_bytes(pages):
    """Create a small real PDF in memory. `pages` is a list of page texts."""
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def sample_pdf():
    return make_pdf_bytes(
        [
            "Machine learning is a field of artificial intelligence where computers learn patterns from data "
            "instead of following fixed rules. A model is trained on examples and then makes predictions.",
            "The main types of machine learning are supervised learning, unsupervised learning and "
            "reinforcement learning. Supervised learning uses labelled data to predict outcomes.",
        ]
    )
