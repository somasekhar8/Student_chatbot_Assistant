import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load the free local Hugging Face embedding model once and reuse it.

    Loading takes a few seconds, so it is cached for the whole process.
    Vectors are normalised, so the dot product equals cosine similarity
    (the retriever relies on this for its relevance score).
    """
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
