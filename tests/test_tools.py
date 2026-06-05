"""Tests for the agent tools (tools.py).

Backends (Tavily, Milvus) are faked, so these run with no network/model/Milvus.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

import tools


# --------------------------------------------------------------------------- #
# internet_search                                                              #
# --------------------------------------------------------------------------- #
class FakeTavily:
    def __init__(self):
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {"results": []}


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset the lazy backend singletons before each test."""
    monkeypatch.setattr(tools, "_vectorstore", None)
    monkeypatch.setattr(tools, "_tavily_client", None)


def test_internet_search_forwards_args(monkeypatch):
    fake = FakeTavily()
    monkeypatch.setattr(tools, "_tavily", lambda: fake)

    tools.internet_search("ai news", max_results=3, topic="news")

    assert fake.calls == [
        {
            "query": "ai news",
            "max_results": 3,
            "topic": "news",
            "include_raw_content": False,
        }
    ]


def test_internet_search_client_built_lazily_once(monkeypatch):
    builds = {"n": 0}

    def fake_client(api_key):
        builds["n"] += 1
        return FakeTavily()

    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(tools, "TavilyClient", fake_client)

    tools.internet_search("a")
    tools.internet_search("b")

    assert builds["n"] == 1


def test_internet_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        tools.internet_search("a")


# --------------------------------------------------------------------------- #
# kb_rag_search                                                               #
# --------------------------------------------------------------------------- #
class FakeStore:
    def __init__(self, results=None, error=None):
        self._results = results or []
        self._error = error
        self.calls = []

    def similarity_search_with_score(self, query, k=5):
        self.calls.append({"query": query, "k": k})
        if self._error is not None:
            raise self._error
        return self._results


def use_store(monkeypatch, store):
    monkeypatch.setattr(tools, "get_vectorstore", lambda: store)
    return store


def test_maps_results_to_dicts(monkeypatch):
    use_store(
        monkeypatch,
        FakeStore([(Document(page_content="body", metadata={"source": "a.md"}), 0.87)]),
    )

    out = tools.kb_rag_search("q")

    assert out == [{"content": "body", "source": "a.md", "score": 0.87}]


def test_forwards_query_and_k(monkeypatch):
    store = use_store(monkeypatch, FakeStore([]))

    tools.kb_rag_search("how to deploy", k=3)

    assert store.calls == [{"query": "how to deploy", "k": 3}]


def test_default_k_is_5(monkeypatch):
    store = use_store(monkeypatch, FakeStore([]))

    tools.kb_rag_search("q")

    assert store.calls[0]["k"] == 5


def test_missing_source_defaults_to_unknown(monkeypatch):
    use_store(
        monkeypatch,
        FakeStore([(Document(page_content="orphan", metadata={}), 0.5)]),
    )

    out = tools.kb_rag_search("q")

    assert out[0]["source"] == "unknown"


def test_score_is_coerced_to_float(monkeypatch):
    # Milvus/numpy may hand back a non-native float; the tool must normalize it.
    use_store(
        monkeypatch,
        FakeStore([(Document(page_content="x", metadata={"source": "s"}), 1)]),
    )

    out = tools.kb_rag_search("q")

    assert isinstance(out[0]["score"], float)
    assert out[0]["score"] == 1.0


def test_vectorstore_built_once_and_reused(monkeypatch):
    build_count = {"n": 0}
    store = FakeStore([])

    def counting_factory():
        build_count["n"] += 1
        return store

    monkeypatch.setattr(tools, "get_vectorstore", counting_factory)

    tools.kb_rag_search("a")
    tools.kb_rag_search("b")
    tools.kb_rag_search("c")

    # Loading the embedding model is expensive — build the store only once.
    assert build_count["n"] == 1


def test_backend_error_returns_readable_message(monkeypatch):
    use_store(monkeypatch, FakeStore(error=ConnectionError("connection refused")))

    # Must not raise — the agent should be able to fall back to web search.
    out = tools.kb_rag_search("q")

    assert isinstance(out, str)
    assert "knowledge base" in out.lower()


def test_backend_error_resets_singleton_for_retry(monkeypatch):
    # A failed call shouldn't keep a broken store; a later call can retry.
    use_store(monkeypatch, FakeStore(error=ConnectionError("down")))
    tools.kb_rag_search("q")

    use_store(
        monkeypatch,
        FakeStore([(Document(page_content="ok", metadata={"source": "s"}), 0.9)]),
    )
    out = tools.kb_rag_search("q")

    assert out == [{"content": "ok", "source": "s", "score": 0.9}]
