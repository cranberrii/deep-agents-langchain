"""
Simple Deep Research Agent
==========================

A CLI deep-research agent built on LangChain's `deepagents` framework.

It uses:
  * An OpenAI-compatible chat endpoint (OpenRouter, vLLM, LM Studio, etc.)
    configured purely through environment variables.
  * Tavily for web search.
  * A private Milvus knowledge base (RAG) via the `kb_rag_search` tool, for
    internal/proprietary documents. Populate it with `ingest.py`.
  * A dedicated research sub-agent that the main agent delegates focused
    sub-questions to (the "deep" part of deep research).

The main agent plans, delegates research to the sub-agent, and writes a final
report into the agent's virtual filesystem, which we then print and save.

Usage
-----
    python research_agent.py "What are the latest advances in solid-state batteries?"

    # or run with no argument and you'll be prompted:
    python research_agent.py

Environment
-----------
    OPENAI_BASE_URL   Base URL of the OpenAI-compatible endpoint.
                      OpenRouter: https://openrouter.ai/api/v1
                      vLLM:       http://localhost:8000/v1
    OPENAI_API_KEY    API key for that endpoint (any non-empty string for vLLM).
    MODEL_NAME        Model id, e.g. "anthropic/claude-sonnet-4-6" (OpenRouter)
                      or the served model name for vLLM.
    TAVILY_API_KEY    API key from https://app.tavily.com
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from tools import internet_search, kb_rag_search, require_env

load_dotenv()

# Knob that shapes the model sampling temperature.
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.0"))

# All agent file I/O is persisted to disk under the project's `runs/` folder.
# With `virtual_mode=True`, the agent's absolute paths (e.g. `/final_report.md`)
# resolve *inside* RUNS_DIR instead of the real filesystem root, and path
# traversal (`..`, `~`) is blocked.
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

RESEARCH_SUBAGENT_PROMPT = f"""You are a focused research specialist. Today is {CURRENT_DATE}.

You are given ONE topic or sub-question at a time. You have two sources:
- `kb_rag_search`: the team's private document knowledge base (internal docs).
- `internet_search`: the public web.

Your job:
1. Check `kb_rag_search` first for internal/proprietary topics; use
   `internet_search` for current public information. Use both when relevant.
2. Cross-check claims across multiple sources before trusting them.
3. Return a concise, factual briefing with key findings. Cite where each finding
   came from — a KB source path for knowledge-base hits, a URL for web hits. Do
   not write the final report — just hand back findings.

Be rigorous: prefer primary sources, note disagreements between sources, and
flag anything uncertain.
"""

MAIN_AGENT_PROMPT = f"""You are an expert research lead. Today is {CURRENT_DATE}.

Your job is to produce a thorough, well-structured research report answering the
user's question.

## Workflow
1. **Plan** the research: break the question into 2-5 focused sub-questions and
   record the plan with the write_todos planning tool.
2. **Delegate** each sub-question to the `research-agent` sub-agent. Give it ONE
   topic at a time. You may delegate several over the course of the task.
3. **Synthesize** the findings into a single coherent report. Resolve conflicts
   between sources and note remaining uncertainty.
4. **Write** the final report to a file named `final_report.md` using the file
   tools, then also return it as your final message.

## Report format
- Start with a 2-4 sentence executive summary.
- Use clear markdown headings for each major section.
- Include a "Sources" section at the end. Distinguish knowledge-base sources
  (file paths) from web sources (URLs).
- Be specific and cite figures/dates where relevant. Avoid filler.

You also have `kb_rag_search` (private knowledge base) and `internet_search`
directly for quick lookups, but prefer delegating substantial research to the
sub-agent. Consult the knowledge base first for internal/proprietary topics.
"""


# --------------------------------------------------------------------------- #
# Sub-agent + main agent                                                       #
# --------------------------------------------------------------------------- #
def build_agent():
    """Build the deep research agent. Reads env config and constructs the model."""
    model = ChatOpenAI(
        model=require_env("MODEL_NAME"),
        base_url=require_env("OPENAI_BASE_URL"),
        api_key=require_env("OPENAI_API_KEY"),
        temperature=MODEL_TEMPERATURE,
    )

    research_subagent = {
        "name": "research-agent",
        "description": (
            "Delegate a focused research topic or sub-question to this agent. "
            "Give it ONE topic at a time; it returns findings with sources."
        ),
        "system_prompt": RESEARCH_SUBAGENT_PROMPT,
        "tools": [internet_search, kb_rag_search],
    }

    # Persist all agent files (reports, summarization offloads, etc.) to disk
    # under the project's `runs/` folder rather than ephemeral agent state.
    os.makedirs(RUNS_DIR, exist_ok=True)
    backend = FilesystemBackend(root_dir=RUNS_DIR, virtual_mode=True)

    return create_deep_agent(
        model=model,
        tools=[internet_search, kb_rag_search],
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=[research_subagent],
        backend=backend,
    )


# --------------------------------------------------------------------------- #
# Streaming / logging                                                          #
# --------------------------------------------------------------------------- #
def _agent_label(namespace: tuple[str, ...]) -> str:
    """Human-readable label for which (sub)agent a stream update came from.

    LangGraph reports subgraph updates with a `namespace` tuple whose entries
    look like ``"<node>:<run-id>"``. An empty tuple is the top-level (main)
    agent; any non-empty namespace means we're inside the `task` tool running a
    sub-agent as a nested subgraph.
    """
    if not namespace:
        return "main-agent"
    nodes = [segment.split(":", 1)[0] for segment in namespace]
    return "subagent:" + " > ".join(nodes)


def _log_message(message, label: str, indent: str) -> None:
    """Print one message plus any reasoning/tool-call detail it carries."""
    print(f"\n{indent}┌─[{label}] {message.__class__.__name__}")

    # Surface "thinking" — reasoning models stash it in additional_kwargs and
    # different OpenAI-compatible backends use different key names.
    extra = getattr(message, "additional_kwargs", {}) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning")
    if reasoning:
        print(f"{indent}│ 🧠 thinking: {reasoning}")

    # Tool calls the agent decided to make this step.
    for call in getattr(message, "tool_calls", None) or []:
        print(f"{indent}│ 🔧 tool_call: {call.get('name')}({call.get('args')})")

    message.pretty_print()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def get_question() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:]).strip()
    try:
        return input("Research question: ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nNo question provided.")


def main() -> None:
    question = get_question()
    if not question:
        sys.exit("No question provided.")

    print(f"\n🔎 Researching: {question}\n" + "=" * 70)

    agent = build_agent()
    # Stream intermediate steps so you can watch planning + delegation happen.
    # `subgraphs=True` makes LangGraph emit updates from *inside* the `task`
    # tool's nested sub-agent runs too — so we see the sub-agent's own thinking,
    # tool calls and responses instead of just the collapsed task result.
    for namespace, step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
        subgraphs=True,
    ):
        label = _agent_label(namespace)
        indent = "    " * len(namespace)
        for node, update in step.items():
            if not update:
                continue
            messages = update.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    _log_message(message, f"{label}/{node}", indent)

    # The agent writes the report straight to disk via FilesystemBackend, so we just read it back from the project dir.
    report_path = os.path.join(RUNS_DIR, "final_report.md")
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as fh:
            report = fh.read()
        print("\n" + "=" * 70)
        print(f"✅ Report written to {report_path}\n")
        print(report)


if __name__ == "__main__":
    main()
