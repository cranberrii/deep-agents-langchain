"""Tests for research_agent wiring.

Verifies the agent is assembled with the expected tools, without building a real
model or deep agent (those are mocked).
"""

from __future__ import annotations

import research_agent


def test_build_agent_registers_both_tools_on_main_and_subagent(monkeypatch):
    captured = {}
    monkeypatch.setenv("OPENAI_BASE_URL", "http://endpoint/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("MODEL_NAME", "some-model")
    monkeypatch.setattr(research_agent, "ChatOpenAI", lambda **kw: "MODEL")
    monkeypatch.setattr(
        research_agent,
        "create_deep_agent",
        lambda **kw: captured.update(kw) or "AGENT",
    )

    agent = research_agent.build_agent()

    assert agent == "AGENT"

    main_tools = {t.__name__ for t in captured["tools"]}
    assert {"internet_search", "kb_rag_search"} <= main_tools

    subagent = captured["subagents"][0]
    sub_tools = {t.__name__ for t in subagent["tools"]}
    assert {"internet_search", "kb_rag_search"} <= sub_tools


def test_build_agent_requires_model_config(monkeypatch):
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with __import__("pytest").raises(SystemExit):
        research_agent.build_agent()
