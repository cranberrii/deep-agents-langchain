# Agent Specs — Milvus Vector DB Retrieval Component

Status: **Implemented**

This document specifies the **Milvus vector database** that serves as a second
knowledge source for the deep research agent. The agent has a tool,
`kb_rag_search`, that retrieves passages from a private document collection —
complementing the existing `internet_search` (Tavily) web tool. Both tools now
live together in `tools.py`.

---

## 1. Decisions (locked)

| Area | Choice | Notes |
|------|--------|-------|
| Deployment | **Milvus Standalone (Docker)** | `docker compose up -d`; agent connects over `localhost:19530`. |
| Embeddings | **Local sentence-transformers** | `BAAI/bge-m3` (1024-dim, multilingual) via `langchain-huggingface`. No API cost, offline. |
| Ingestion | **Separate `ingest.py` script** | Loads `./data/`, chunks, embeds, upserts. Run once before querying. |
| Use case | **Private docs / RAG** | Agent searches local docs *and* the web (hybrid knowledge). |
| Integration | `langchain-milvus` `Milvus` vector store | Clean LangChain interface; works with deepagents tools list. **Pinned `langchain-milvus==0.2.2` + `pymilvus>=2.5.7,<2.6`** (see §9 — 0.3.x/pymilvus 2.6 is broken). |

Embedding model is swappable via env var; **dimension follows the model**
(bge-m3 = 1024; for reference bge-small = 384, bge-base = 768, bge-large = 1024).
Milvus infers dim from the embedding object, so no manual dim config is needed in
the store, but it must stay consistent between ingestion and retrieval.

`bge-m3` is multilingual and supports long inputs (up to 8192 tokens), which
suits mixed-language private docs. Trade-off: larger model (~2.3 GB download,
~570 M params) and slower CPU inference than bge-small.

---

## 2. Target architecture

```
                         ┌───────────────────────────────┐
   user question  ─────► │   Main deep agent             │
                         │   (plans, delegates, writes)  │
                         └───────┬────────────────┬──────┘
                                 │ delegates      │ direct
                                 ▼                ▼
                    ┌─────────────────────┐  tools:
                    │  research-agent     │    • internet_search (Tavily)
                    │  (sub-agent)        │    • kb_rag_search
                    │  tools:             │
                    │  • internet_search  │
                    │  • kb_rag_search
                    └─────────┬───────────┘
                              │
                              ▼
                 ┌──────────────────────────┐      ┌───────────────────┐
                 │ Milvus vector store      │◄────►│ Milvus Standalone │
                 │ (langchain_milvus.Milvus)│      │ (Docker :19530)   │
                 └────────────┬─────────────┘      └───────────────────┘
                              │ embeds queries with
                              ▼
                 ┌─────────────────────────┐
                 │ HuggingFaceEmbeddings   │
                 │ BAAI/bge-m3 (1024-dim)  │
                 └─────────────────────────┘

   Offline / one-time:
   ./data/*  ──►  ingest.py  ──► chunk ──► embed ──► upsert into Milvus collection
```

---

## 3. Files to add / change

| File | Action | Purpose |
|------|--------|---------|
| `docker-compose.yml` | **added** | Run Milvus Standalone + its etcd & MinIO deps locally. |
| `kb.py` | **added** | Shared factory: build embeddings + connect to the Milvus vector store. Single source of truth used by both ingest and the agent. |
| `ingest.py` | **added** | CLI: load `./data/`, chunk, embed, upsert into the collection. |
| `tools.py` | **added** | Both agent tools — `internet_search` (Tavily) and `kb_rag_search` (Milvus RAG) — with lazily-built backends so import is side-effect-free. |
| `research_agent.py` | **modified** | Imports both tools from `tools.py`; registers them on main agent + sub-agent; prompts updated. Agent construction moved into `build_agent()` so import needs no env (testability). |
| `pyproject.toml` | **modified** | Deps: pinned `langchain-milvus==0.2.2`, `pymilvus>=2.5.7,<2.6`, plus `langchain-huggingface`, `sentence-transformers`, `langchain-text-splitters`, `langchain-community`, `pypdf`. Dev group: `pytest`. |
| `.env.example` | **modified** | Add Milvus + embedding config vars. |
| `data/` | **added (dir)** | Drop source documents here (`.md`, `.txt`, `.pdf`). Git-ignored. |
| `tests/` | **added** | `test_kb.py`, `test_ingest.py`, `test_tools.py`, `test_research_agent.py`. |
| `README.md` | **modified** | Document Milvus setup + ingestion step. |
| `.gitignore` | **added** | Ignore `volumes/` (Milvus data), `data/`, `.env`, `.venv/`, `final_report.md`. |

---

## 4. Component specs

### 4.1 `docker-compose.yml` — Milvus Standalone

Use the official Milvus standalone compose (Milvus + etcd + MinIO). Key points:

- Service `standalone` exposes **`19530`** (gRPC) and `9091` (health/metrics).
- Persists to `./volumes/` (add to `.gitignore`).
- Health: `curl -f http://localhost:9091/healthz` before the agent connects.

Bring-up:
```bash
docker compose up -d
# wait until: docker compose ps  -> standalone "healthy"
```

### 4.2 `kb.py` — shared embeddings + vector store factory

Single module both `ingest.py` and `research_agent.py` import, so embedding model
and connection settings never drift.

```python
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION = os.environ.get("MILVUS_COLLECTION", "research_kb")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")

def get_embeddings() -> HuggingFaceEmbeddings:
    # normalize_embeddings=True -> cosine-friendly vectors
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore(drop_old: bool = False) -> Milvus:
    return Milvus(
        embedding_function=get_embeddings(),
        collection_name=COLLECTION,
        connection_args={"uri": MILVUS_URI},
        index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX"},
        auto_id=True,
        drop_old=drop_old,
    )
```

### 4.3 `ingest.py` — ingestion pipeline

Responsibilities:
1. Discover files under `./data/` (`.md`, `.txt`, `.pdf`).
2. Load: `TextLoader` for text/markdown, `PyPDFLoader` for PDFs
   (from `langchain_community.document_loaders`).
3. Chunk: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`.
4. Attach metadata: `{"source": <relative path>}` (used for citations).
5. Upsert: `Milvus.from_documents(...)` with `drop_old=True` on first build, or
   `vectorstore.add_documents(chunks)` for incremental adds.

Sketch:
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from kb import get_embeddings, MILVUS_URI, COLLECTION
from langchain_milvus import Milvus

# ...load + split into `chunks`...
Milvus.from_documents(
    documents=chunks,
    embedding=get_embeddings(),
    collection_name=COLLECTION,
    connection_args={"uri": MILVUS_URI},
    index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX"},
    drop_old=True,   # rebuild collection from scratch
)
```
CLI: `uv run ingest.py [path]` (defaults to `./data`). Prints counts:
files found, chunks created, vectors written.

### 4.4 `kb_rag_search` tool — in `tools.py`

Both tools live in `tools.py`. `kb_rag_search` uses a lazy singleton for the
vector store (loading the embedding model is expensive) and degrades gracefully
if Milvus is down — returning a short message string instead of raising, so the
agent can fall back to web search.

```python
from kb import get_vectorstore

_vectorstore = None  # lazy singleton (avoids loading the model at import time)

def _kb():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore()
    return _vectorstore

def kb_rag_search(query: str, k: int = 5) -> list[dict] | str:
    """Search the private document knowledge base (RAG) for relevant passages.

    Use this for internal/private/domain-specific information that may not be on
    the public web. Returns the most relevant chunks with their source and a
    relevance score; returns a message string if the KB is unavailable.
    """
    global _vectorstore
    try:
        docs_and_scores = _kb().similarity_search_with_score(query, k=k)
    except Exception as exc:        # e.g. Milvus down — degrade gracefully
        _vectorstore = None         # drop broken store so a later call retries
        return f"The knowledge base is currently unavailable... Error: {exc}"
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": float(score),
        }
        for doc, score in docs_and_scores
    ]
```

Registration (in `research_agent.build_agent()`):
- `kb_rag_search` is in the main agent's `tools=[...]`.
- It is also in `research_subagent["tools"]` so delegated research can hit it.

### 4.5 Prompt updates

- **Main agent prompt:** add a tool description and guidance — "Prefer
  `kb_rag_search` for internal/proprietary topics; use `internet_search`
  for current public information. When both are relevant, consult the knowledge
  base first, then fill gaps with the web."
- **Sub-agent prompt:** note it now has two sources and should cite which one
  each finding came from (KB source path vs. URL).
- Report's **Sources** section: distinguish KB sources (file paths) from web URLs.

---

## 5. Configuration (`.env.example` additions)

```bash
# --- Milvus vector DB -----------------------------------------------------
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=research_kb

# --- Embeddings (local, no API cost) --------------------------------------
EMBED_MODEL=BAAI/bge-m3   # 1024-dim, multilingual, long-context
```

No API key needed for embeddings (runs locally). First run downloads the model
(~2.3 GB for bge-m3) from HuggingFace.

---

## 6. Dependencies (`pyproject.toml`)

Add to `dependencies`:
```
"langchain-milvus",
"pymilvus",
"langchain-huggingface",
"sentence-transformers",
"langchain-text-splitters",
"langchain-community",
"pypdf",                 # only if ingesting PDFs
```
Then `uv sync`. Note: `sentence-transformers` pulls in `torch` (large download).

---

## 7. Build sequence

1. **Add deps** → `uv sync`.
2. **Write `docker-compose.yml`** → `docker compose up -d` → wait for healthy.
3. **Write `kb.py`** (shared factory).
4. **Write `ingest.py`**; drop sample docs in `./data/`; run `uv run ingest.py`.
   Verify vector count > 0.
5. **Smoke-test retrieval** standalone:
   `uv run python -c "from kb import get_vectorstore; print(get_vectorstore().similarity_search('test', k=2))"`
6. **Add the tools** to `tools.py`; register them in `research_agent.build_agent()`
   (main + sub-agent) and update prompts.
7. **End-to-end run** with a question answerable only from `./data/` to confirm
   the agent actually pulls from the KB (check it cites a file path).

---

## 8. Testing & verification

- **Ingestion:** assert reported chunk count matches expectation; re-running with
  `drop_old=True` should not duplicate.
- **Retrieval relevance:** query with a phrase known to be in a doc; top result's
  `source` should be that file, score within expected range (COSINE: higher =
  closer; note langchain may return distance — verify ordering empirically).
- **Tool isolation:** unit-test `kb_rag_search` returns the dict shape
  `{content, source, score}`.
- **Agent integration:** ask a private-only question → report cites a KB path;
  ask a current-events question → report uses web URLs. Confirms routing.
- **Failure modes** (see §9) are surfaced as readable messages, not stack traces.

---

## 9. Edge cases & risks

| Risk | Mitigation |
|------|-----------|
| **`langchain-milvus` 0.3.x + `pymilvus` 2.6 is broken** | pymilvus 2.6's `MilvusClient` no longer registers an ORM connection, but `langchain-milvus` 0.3.3's `col` path still uses the ORM `Collection` → `ConnectionNotExistException` on every insert/search. **Pinned `langchain-milvus==0.2.2` + `pymilvus>=2.5.7,<2.6`** (the working pair). Revisit when a fixed `langchain-milvus` ships. |
| Milvus not running / unreachable | Tool catches connection errors, returns a clear message ("knowledge base unavailable — is Milvus up?") so the agent can fall back to web search instead of crashing. |
| Empty collection (no ingest yet) | `ingest.py` warns if `./data/` is empty; tool returns `[]` gracefully. |
| Embedding model loads on every call | Lazy singleton (`_kb()`) loads once per process. |
| Dimension mismatch after changing `EMBED_MODEL` | Re-run `ingest.py` with `drop_old=True`; document this in README. Mismatched dim → Milvus insert/search error. |
| `torch` install size / no GPU | bge-m3 runs on CPU but is heavier (~2.3 GB download, slower than bge-small). Acceptable for small KBs; consider a GPU or batching for large ingests. Document the first-run model download. |
| Score semantics (distance vs. similarity) | Normalize embeddings + COSINE; verify ordering during testing and document which direction is "better". |
| Cold-start latency in agent | Optionally pre-warm `_kb()` at startup so the first tool call isn't slow. |

---

## 10. Future extensions 

- **Hybrid search:** add `BM25BuiltInFunction` for dense+sparse retrieval
  (`vector_field=["dense", "sparse"]`) — improves keyword recall.
- **Metadata filtering:** filter retrieval by source/date via Milvus `expr`.
- **Reranking:** add a cross-encoder reranker over top-k for higher precision.
- **Agent-writes-to-KB:** add an `add_to_knowledge_base` tool to store research
  findings as new vectors (the "research memory" pattern).
