"""Agent tools.

Two tools the research agent uses to gather information:
  * `internet_search` — public web search via Tavily.
  * `kb_rag_search`   — private document knowledge base (Milvus RAG).

Both lazily initialize their backends so importing this module is side-effect
free (no network, no model load, no env required until a tool actually runs).
"""

from __future__ import annotations

import os
import sys
from typing import Literal

from tavily import TavilyClient

from kb import get_vectorstore

MAX_SEARCH_RESULTS = int(os.environ.get("MAX_SEARCH_RESULTS", "5"))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(
            f"Missing required environment variable: {name}\n"
            "Copy .env.example to .env and fill it in (or export it)."
        )
    return value


# --------------------------------------------------------------------------- #
# Web search via Tavily                                                        #
# --------------------------------------------------------------------------- #
_tavily_client = None


def _tavily() -> TavilyClient:
    """Lazily build the Tavily client (keeps import side-effect-free)."""
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=require_env("TAVILY_API_KEY"))
    return _tavily_client


def internet_search(
    query: str,
    max_results: int = MAX_SEARCH_RESULTS,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search and return ranked results.

    Args:
        query: The search query.
        max_results: How many results to return.
        topic: Search topic bias ("general", "news", or "finance").
        include_raw_content: Whether to include the scraped page text.
    """
    return _tavily().search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


# --------------------------------------------------------------------------- #
# Private knowledge base retrieval (Milvus RAG)                                #
# --------------------------------------------------------------------------- #
_vectorstore = None


def _kb():
    """Lazily build the vector store once (loading the model is expensive)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore


def kb_rag_search(query: str, k: int = 5) -> list[dict] | str:
    """Search the private document knowledge base (RAG) for relevant passages.

    Use this for internal, proprietary, or domain-specific information that may
    not be on the public web. Prefer this over `internet_search` for questions
    about private docs; use both when a topic spans internal and public sources.

    Args:
        query: Natural-language search query.
        k: Number of passages to return (default 5).

    Returns:
        A list of {"content", "source", "score"} dicts, best match first. If the
        knowledge base is unavailable, returns a short message string instead.
    """
    global _vectorstore
    try:
        docs_and_scores = _kb().similarity_search_with_score(query, k=k)
    except Exception as exc:  # e.g. Milvus down — degrade gracefully
        # Drop the (possibly broken) cached store so a later call can retry.
        _vectorstore = None
        return (
            "The knowledge base is currently unavailable (is Milvus running?). "
            f"Fall back to other tools. Error: {exc}"
        )
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
        }
        for doc, score in docs_and_scores
    ]
