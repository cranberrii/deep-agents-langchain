"""
Knowledge base factory — shared by `ingest.py` and `research_agent.py`.

Single source of truth for the embedding model and Milvus connection so the two
never drift (a dimension/connection mismatch between ingestion and retrieval is
the easiest way to break RAG).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

load_dotenv()

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.environ.get("MILVUS_COLLECTION", "research_kb")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")  # 1024-dim, multilingual

# COSINE pairs with normalized embeddings; AUTOINDEX lets Milvus pick a good index.
INDEX_PARAMS = {"metric_type": "COSINE", "index_type": "AUTOINDEX"}


def get_embeddings() -> HuggingFaceEmbeddings:
    """Local sentence-transformers embeddings (no API cost).

    First call downloads the model (~2.3 GB for bge-m3) from HuggingFace.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(
    embeddings: HuggingFaceEmbeddings | None = None,
    drop_old: bool = False,
) -> Milvus:
    """Connect to (or create) the Milvus collection as a LangChain vector store.

    Args:
        embeddings: Reuse an existing embeddings object (avoids reloading the
            model). If None, a fresh one is built.
        drop_old: Drop and recreate the collection if it already exists.
    """
    return Milvus(
        embedding_function=embeddings or get_embeddings(),
        collection_name=COLLECTION,
        connection_args={"uri": MILVUS_URI},
        index_params=INDEX_PARAMS,
        auto_id=True,
        drop_old=drop_old,
    )
