# AGENT.md

Guidance for AI agents working in this repo. See `README.md` for user-facing
docs and `agent_specs.md` for the Milvus/RAG design.

## What this is

A CLI deep-research agent built on LangChain `deepagents`. It plans, delegates to
a research sub-agent, and writes `final_report.md`. Two information sources:
`internet_search` (Tavily) and `kb_rag_search` (private Milvus RAG, `bge-m3`
embeddings).

## Layout

| File | Purpose |
|------|---------|
| `research_agent.py` | CLI entry; `build_agent()` assembles the deep agent |
| `tools.py` | Both agent tools (`internet_search`, `kb_rag_search`) |
| `kb.py` | Shared embeddings + Milvus vector-store factory |
| `ingest.py` | Loads `./data/` into Milvus |
| `docker-compose.yml` | Milvus Standalone (+ etcd, MinIO) |
| `tests/` | pytest suite |

## Commands

```bash
uv sync                             # install deps (incl. dev group)
uv run pytest                       # full suite (needs Milvus for 1 integration test)
uv run pytest -m "not integration"  # fast unit tests only
docker compose up -d                # start Milvus on :19530
uv run ingest.py                    # (re)build the knowledge base from ./data
uv run research_agent.py "..."      # run the agent
```

Use `uv run ...`; never call `pip` or a bare `python`.

## Conventions

- **TDD.** Write a failing test first, then minimal code. Tests mock heavy
  backends (the embedding model, Milvus, Tavily) so units run with no
  network/GPU/Milvus.
- **Side-effect-free imports.** No env reads that `sys.exit`, no model loads, no
  network at import time. Defer to functions (`build_agent`, `_kb`, `_tavily`)
  and `require_env(...)`. This keeps everything importable in tests.
- **Lazy singletons** for expensive backends; reset on error so a later call can
  retry (see `kb_rag_search`).
- **Tools degrade gracefully** — return a readable message string instead of
  raising, so the agent can fall back.
- Match surrounding style; keep comments sparse and high-signal.

## Gotchas

- **Pinned `langchain-milvus==0.2.2` + `pymilvus>=2.5.7,<2.6`.** Do NOT bump to
  `langchain-milvus` 0.3.x / `pymilvus` 2.6 — they're incompatible
  (`ConnectionNotExistException` on every op). See `agent_specs.md` §9.
- `EMBED_MODEL` dimension must match between ingest and retrieval; changing it
  requires re-running `ingest.py`.
- `data/` and `volumes/` are git-ignored (private docs + Milvus data).
