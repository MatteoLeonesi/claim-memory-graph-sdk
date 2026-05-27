from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from cmg.integration import AsyncLLMFn, AsyncLLMStreamFn, Message


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(system_parts), rest


def make_anthropic_llm_fn(
    model: str,
    *,
    max_tokens: int = 1024,
    **client_kwargs: Any,
) -> AsyncLLMFn:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ImportError(
            "install claim-memory-graph[anthropic] to use the anthropic provider"
        ) from exc

    client = AsyncAnthropic(**client_kwargs)

    async def llm_fn(messages: list[Message]) -> str:
        system, rest = _split_system(messages)
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system or None,
            messages=rest,
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts)

    return llm_fn


def make_anthropic_astream_fn(
    model: str,
    *,
    max_tokens: int = 1024,
    **client_kwargs: Any,
) -> AsyncLLMStreamFn:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ImportError(
            "install claim-memory-graph[anthropic] to use the anthropic provider"
        ) from exc

    client = AsyncAnthropic(**client_kwargs)

    async def stream_fn(messages: list[Message]) -> AsyncIterator[str]:
        system, rest = _split_system(messages)
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system or None,
            messages=rest,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    return stream_fn


__all__ = ["make_anthropic_astream_fn", "make_anthropic_llm_fn"]
