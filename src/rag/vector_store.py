"""One FAISS index per session, kept in memory.

The original project saved a single global index to `.faiss_index/`, which every
visitor shared. Here each session id owns a separate index, so students can
never retrieve each other's documents.

Limitations (kept simple on purpose): indexes live in the memory of one server
process, disappear on restart, and are not shared between several workers.
Old sessions are dropped once MAX_SESSIONS is reached.
"""
import threading
from collections import OrderedDict
from typing import Dict, List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document

from src.rag.embeddings import get_embedding_model

MAX_SESSIONS = 100


class SessionIndexStore:
    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._max_sessions = max_sessions
        self._indexes: "OrderedDict[str, FAISS]" = OrderedDict()
        self._files: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def add_documents(self, session_id: str, chunks: List[Document], filename: str) -> bool:
        """Index chunks for a session. Returns False if the file was already indexed."""
        if not chunks:
            return False
        with self._lock:
            if filename in self._files.get(session_id, []):
                return False
            embeddings = get_embedding_model()
            if session_id in self._indexes:
                self._indexes[session_id].add_documents(chunks)
                self._indexes.move_to_end(session_id)
            else:
                # Inner product on normalised vectors == cosine similarity (higher is better).
                self._indexes[session_id] = FAISS.from_documents(
                    chunks, embeddings, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                )
                self._files[session_id] = []
            self._files[session_id].append(filename)
            self._evict_old_sessions()
            return True

    def _evict_old_sessions(self) -> None:
        while len(self._indexes) > self._max_sessions:
            oldest, _ = self._indexes.popitem(last=False)
            self._files.pop(oldest, None)

    def has_documents(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._indexes

    def list_files(self, session_id: str) -> List[str]:
        with self._lock:
            return list(self._files.get(session_id, []))

    def search(self, session_id: str, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Return (chunk, cosine similarity) pairs, best first. Empty if the session has no index."""
        with self._lock:
            index = self._indexes.get(session_id)
        if index is None:
            return []
        results = index.similarity_search_with_score(query, k=k)
        return [(doc, float(score)) for doc, score in results]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._indexes.pop(session_id, None)
            self._files.pop(session_id, None)


_store = SessionIndexStore()


def get_index_store() -> SessionIndexStore:
    """The shared store used by the whole application."""
    return _store
