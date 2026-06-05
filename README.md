# Basic Deep Research Agent

A minimal CLI deep-research agent built on LangChain's
[`deepagents`](https://docs.langchain.com/oss/python/deepagents) framework.

The main agent **plans** the research, **delegates** focused sub-questions to a
dedicated research **sub-agent**, then **synthesizes** a structured report and
writes it to `final_report.md`. It draws on two sources:

- **`internet_search`** — public web via Tavily.
- **`kb_rag_search`** — a private document knowledge base (RAG) backed by
  **Milvus**, embedded locally with `BAAI/bge-m3`. Populate it with `ingest.py`.

Works with any **OpenAI-compatible endpoint** — OpenRouter, vLLM, LM Studio,
etc. — configured entirely through environment variables.

## Setup

This project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # creates .venv and installs dependencies
cp .env.example .env   # then edit .env with your keys
```

You need:
- An OpenAI-compatible chat endpoint (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_NAME`)
- A [Tavily](https://app.tavily.com) key (`TAVILY_API_KEY`)
- Docker (for the Milvus knowledge base)

## Knowledge base (Milvus + RAG)

Start Milvus locally and load your private documents before running the agent.

```bash
docker compose up -d              # start Milvus (wait until `docker compose ps` shows healthy)

mkdir -p data                     # drop .md / .txt / .pdf files into ./data
uv run ingest.py                  # chunk, embed (bge-m3), and upsert into Milvus
# uv run ingest.py --append       # add to the existing collection without rebuilding
```

First ingest downloads the `bge-m3` embedding model (~2.3 GB). Re-run `ingest.py`
whenever your docs change. If you switch `EMBED_MODEL`, re-run it (the vector
dimension must match between ingestion and retrieval).

Stop Milvus with `docker compose down` (add `-v` to also wipe `./volumes/`).

## Usage

```bash
uv run research_agent.py "What are the latest advances in solid-state batteries?"

# or via the installed entry point:
uv run research-agent "What are the latest advances in solid-state batteries?"
```

Or run with no argument to be prompted. The final report prints to the terminal
and is saved to `final_report.md`.

## How it works

| Piece | Role |
|-------|------|
| `internet_search` (`tools.py`) | Tavily-backed web search tool |
| `kb_rag_search` (`tools.py`) | Retrieves passages from the private Milvus knowledge base |
| `research-agent` (sub-agent) | Researches one sub-question at a time, returns findings + sources |
| Main agent | Plans → delegates → synthesizes → writes `final_report.md` |

The planning tool (`write_todos`) and a virtual filesystem come built in with
`create_deep_agent`. The agent prefers the knowledge base for internal/proprietary
topics and the web for current public information; if Milvus is down,
`kb_rag_search` returns a message and the agent falls back to web search.

## Project layout

| File | Purpose |
|------|---------|
| `research_agent.py` | CLI entry point; `build_agent()` assembles the deep agent |
| `tools.py` | The two agent tools (web search + KB RAG) |
| `kb.py` | Shared embeddings + Milvus vector-store factory |
| `ingest.py` | Loads `./data/` into the Milvus collection |
| `docker-compose.yml` | Milvus Standalone (+ etcd, MinIO) |
| `tests/` | `pytest` suite |

## Tests

```bash
uv run pytest                      # full suite
uv run pytest -m "not integration" # skip the test that needs a running Milvus
```

## Switching providers

Just change the three `OPENAI_*` env vars. For example, point `OPENAI_BASE_URL`
at `http://localhost:8000/v1` for a local vLLM server, or at
`https://openrouter.ai/api/v1` for OpenRouter.

> **Dependency note:** `langchain-milvus` and `pymilvus` are pinned
> (`==0.2.2` / `>=2.5.7,<2.6`). The newer `langchain-milvus` 0.3.x is broken
> against `pymilvus` 2.6 (it raises `ConnectionNotExistException` on every
> insert/search). See `agent_specs.md` §9.
