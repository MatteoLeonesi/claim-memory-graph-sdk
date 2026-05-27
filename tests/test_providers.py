from __future__ import annotations

import importlib
import sys

import pytest


def test_openai_provider_informative_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)
    mod = importlib.import_module("cmg.providers.openai_provider")
    with pytest.raises(ImportError, match=r"claim-memory-graph\[openai\]"):
        mod.make_openai_llm_fn("gpt-4")


def test_anthropic_provider_informative_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    mod = importlib.import_module("cmg.providers.anthropic_provider")
    with pytest.raises(ImportError, match=r"claim-memory-graph\[anthropic\]"):
        mod.make_anthropic_llm_fn("claude-sonnet-4")


def test_provider_package_exports_helpers() -> None:
    mod = importlib.import_module("cmg.providers")
    assert hasattr(mod, "make_openai_llm_fn")
    assert hasattr(mod, "make_openai_astream_fn")
    assert hasattr(mod, "make_anthropic_llm_fn")
    assert hasattr(mod, "make_anthropic_astream_fn")
