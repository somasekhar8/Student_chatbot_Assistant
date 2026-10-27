"""Retrieval with a relevance check (the first line of defence against hallucination).

Chunks whose similarity score is below RELEVANCE_THRESHOLD are thrown away.
If nothing is left, the chatbot answers with the fixed "not found" message
instead of letting the LLM guess.

RELEVANCE_THRESHOLD is a cosine similarity (0 to 1) for the MiniLM model.
0.30 is a reasonable start; raise it for stricter answers, lower it if good
questions are being rejected.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.rag.vector_store import get_index_store

DEFAULT_TOP_K = 4
DEFAULT_RELEVANCE_THRESHOLD = 0.30


def relevance_threshold() -> float:
    try:
        return float(os.getenv("RELEVANCE_THRESHOLD", DEFAULT_RELEVANCE_THRESHOLD))
    except ValueError:
        return DEFAULT_RELEVANCE_THRESHOLD


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float

    def to_dict(self) -> Dict:
        return {"text": self.text, "document": self.source, "page": self.page, "score": round(self.score, 4)}


@dataclass
class RetrievalResult:
    has_documents: bool
    chunks: List[RetrievedChunk] = field(default_factory=list)  # relevant chunks only
    best_score: Optional[float] = None  # best score before filtering (useful for debugging)

    @property
    def found_relevant(self) -> bool:
        return bool(self.chunks)


def retrieve(
    session_id: str,
    query: str,
    k: int = DEFAULT_TOP_K,
    min_score: Optional[float] = None,
) -> RetrievalResult:
    """Search the session's documents and keep only chunks above the threshold."""
    store = get_index_store()
    if not store.has_documents(session_id):
        return RetrievalResult(has_documents=False)

    threshold = relevance_threshold() if min_score is None else min_score
    raw = store.search(session_id, query, k=k)
    best = raw[0][1] if raw else None

    chunks = [
        RetrievedChunk(
            text=doc.page_content,
            source=str(doc.metadata.get("source", "document")),
            page=int(doc.metadata.get("page", 1)),
            score=score,
        )
        for doc, score in raw
        if score >= threshold
    ]
    return RetrievalResult(has_documents=True, chunks=chunks, best_score=best)
