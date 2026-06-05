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
from tools import internet_search, kb_rag_search, require_env

load_dotenv()

# Knob that shapes the model sampling temperature.
MODEL_TEMPERATURE = float(os.environ.get("MODEL_TEMPERATURE", "0.0"))


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

    return create_deep_agent(
        model=model,
        tools=[internet_search, kb_rag_search],
        system_prompt=MAIN_AGENT_PROMPT,
        subagents=[research_subagent],
    )


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
    final_state = None
    # Stream intermediate steps so you can watch planning + delegation happen.
    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        for _, update in step.items():
            if not update:
                continue
            messages = update.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    message.pretty_print()
            final_state = update

    # Pull the written report out of the agent's virtual filesystem if present.
    report = None
    if final_state and isinstance(final_state.get("files"), dict):
        report = final_state["files"].get("final_report.md")
        # Files may be stored as objects with a "content" attribute/key.
        if report is not None and not isinstance(report, str):
            report = getattr(report, "content", None) or report.get("content")

    if report:
        out_path = "final_report.md"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(report)
        print("\n" + "=" * 70)
        print(f"✅ Report written to {out_path}\n")
        print(report)


if __name__ == "__main__":
    main()
