from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from cmg.integration import AsyncLLMFn, AsyncLLMStreamFn, Message


def make_openai_llm_fn(model: str, **client_kwargs: Any) -> AsyncLLMFn:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "install claim-memory-graph[openai] to use the openai provider"
        ) from exc

    client = AsyncOpenAI(**client_kwargs)

    async def llm_fn(messages: list[Message]) -> str:
        resp = await client.chat.completions.create(model=model, messages=messages)
        content = resp.choices[0].message.content
        return content or ""

    return llm_fn


def make_openai_astream_fn(model: str, **client_kwargs: Any) -> AsyncLLMStreamFn:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "install claim-memory-graph[openai] to use the openai provider"
        ) from exc

    client = AsyncOpenAI(**client_kwargs)

    async def stream_fn(messages: list[Message]) -> AsyncIterator[str]:
        stream = await client.chat.completions.create(model=model, messages=messages, stream=True)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return stream_fn


__all__ = ["make_openai_astream_fn", "make_openai_llm_fn"]
