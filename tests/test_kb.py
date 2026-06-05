"""Unit tests for kb.py.

These mock out the heavy bits (the bge-m3 embedding model and the Milvus
connection) so they run fast with no network, GPU, or running Milvus. One
integration test at the bottom exercises a real round-trip and is skipped when
Milvus is unreachable.
"""

from __future__ import annotations

import importlib
import pytest

import kb


@pytest.fixture(autouse=True)
def _reload_kb():
    """Reload kb after each test so monkeypatched module attrs don't leak."""
    yield
    importlib.reload(kb)


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
def reload_with_env(monkeypatch, **env):
    for key in ("MILVUS_URI", "MILVUS_COLLECTION", "EMBED_MODEL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(kb)


def test_default_config(monkeypatch):
    mod = reload_with_env(monkeypatch)
    assert mod.MILVUS_URI == "http://localhost:19530"
    assert mod.COLLECTION == "research_kb"
    assert mod.EMBED_MODEL == "BAAI/bge-m3"
    assert mod.INDEX_PARAMS == {"metric_type": "COSINE", "index_type": "AUTOINDEX"}


def test_env_overrides(monkeypatch):
    mod = reload_with_env(
        monkeypatch,
        MILVUS_URI="http://milvus.internal:19530",
        MILVUS_COLLECTION="custom_kb",
        EMBED_MODEL="BAAI/bge-small-en-v1.5",
    )
    assert mod.MILVUS_URI == "http://milvus.internal:19530"
    assert mod.COLLECTION == "custom_kb"
    assert mod.EMBED_MODEL == "BAAI/bge-small-en-v1.5"


# --------------------------------------------------------------------------- #
# get_embeddings                                                               #
# --------------------------------------------------------------------------- #
def test_get_embeddings_config(monkeypatch):
    calls = {}

    def fake_embeddings(**kwargs):
        calls.update(kwargs)
        return "FAKE_EMBEDDINGS"

    monkeypatch.setattr(kb, "HuggingFaceEmbeddings", fake_embeddings)

    result = kb.get_embeddings()

    assert result == "FAKE_EMBEDDINGS"
    assert calls["model_name"] == kb.EMBED_MODEL
    # Normalized vectors are required for the COSINE metric to behave.
    assert calls["encode_kwargs"] == {"normalize_embeddings": True}


# --------------------------------------------------------------------------- #
# get_vectorstore                                                              #
# --------------------------------------------------------------------------- #
def test_get_vectorstore_passes_expected_args(monkeypatch):
    captured = {}

    def fake_milvus(**kwargs):
        captured.update(kwargs)
        return "FAKE_STORE"

    monkeypatch.setattr(kb, "Milvus", fake_milvus)
    monkeypatch.setattr(kb, "get_embeddings", lambda: "BUILT_EMBEDDINGS")

    store = kb.get_vectorstore()

    assert store == "FAKE_STORE"
    assert captured["embedding_function"] == "BUILT_EMBEDDINGS"
    assert captured["collection_name"] == kb.COLLECTION
    assert captured["connection_args"] == {"uri": kb.MILVUS_URI}
    assert captured["index_params"] == kb.INDEX_PARAMS
    assert captured["auto_id"] is True
    assert captured["drop_old"] is False


def test_get_vectorstore_reuses_supplied_embeddings(monkeypatch):
    captured = {}
    monkeypatch.setattr(kb, "Milvus", lambda **kw: captured.update(kw))

    def boom():  # get_embeddings must NOT be called when embeddings are supplied
        raise AssertionError("should not build embeddings when one is provided")

    monkeypatch.setattr(kb, "get_embeddings", boom)

    kb.get_vectorstore(embeddings="SUPPLIED", drop_old=True)

    assert captured["embedding_function"] == "SUPPLIED"
    assert captured["drop_old"] is True


# --------------------------------------------------------------------------- #
# Integration (real Milvus) — skipped if unreachable                          #
# --------------------------------------------------------------------------- #
def _milvus_up() -> bool:
    try:
        from pymilvus import MilvusClient

        MilvusClient(uri=kb.MILVUS_URI).list_collections()
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _milvus_up(), reason="Milvus not reachable on MILVUS_URI")
def test_round_trip_against_real_milvus():
    from langchain_core.documents import Document

    collection = "kb_pytest_tmp"
    embeddings = kb.get_embeddings()
    store = kb.get_vectorstore(embeddings=embeddings, drop_old=True)
    # Point this throwaway store at a temp collection to avoid touching real data.
    store = kb.Milvus(
        embedding_function=embeddings,
        collection_name=collection,
        connection_args={"uri": kb.MILVUS_URI},
        index_params=kb.INDEX_PARAMS,
        auto_id=True,
        drop_old=True,
    )
    try:
        store.add_documents(
            [Document(page_content="the sky is blue", metadata={"source": "t"})]
        )
        hits = store.similarity_search("what color is the sky", k=1)
        assert hits and hits[0].metadata["source"] == "t"
    finally:
        store.client.drop_collection(collection)
