"""
llm/generator.py
Generates grounded answers using Gemini.
"""

from __future__ import annotations

from openai import OpenAI
from loguru import logger
import os
from utils import get_env

MODEL = "meta-llama/llama-3.3-70b-instruct"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based strictly on the provided context.

If the answer is not contained within the context, say:
"I don't have enough information to answer that."

Always cite the source document and page number when available.

Be concise, accurate, and well-structured.
"""


def generate_answer(
    query: str,
    context_chunks: list[dict],
    *,
    stream: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> str:

    if not context_chunks:
        logger.warning("No context chunks supplied.")

    prompt = _build_prompt(query, context_chunks)

    model = _get_model()

    logger.info(f"Calling {MODEL}")

    if stream:
        return _stream_response(model, prompt)

    return _blocking_response(model, prompt)


def generate_answer_streaming(
    query: str,
    context_chunks: list[dict],
    max_tokens: int = MAX_TOKENS,
):

    prompt = _build_prompt(query, context_chunks)

    client = _get_model()

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=max_tokens,
    )

    for chunk in stream:

        if chunk.choices:
            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

def _get_model():

    api_key = get_env("OPENROUTER_API_KEY")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def _build_prompt(query: str, chunks: list[dict]) -> str:

    context_parts = []

    for i, chunk in enumerate(chunks, start=1):

        source = chunk.get("source", "unknown")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "")
        score = chunk.get("score")

        score_str = (
            f" (relevance: {score:.3f})"
            if score is not None
            else ""
        )

        context_parts.append(
            f"[{i}] Source: {source}, Page {page}{score_str}\n{text}"
        )

    context_str = "\n\n---\n\n".join(context_parts)

    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer (cite [N] where N is the context number above):"
    )


def _blocking_response(client, prompt):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
    )

    answer = response.choices[0].message.content

    logger.success(
        f"Generated answer ({len(answer)} chars)"
    )

    return answer

def _stream_response(client, prompt):

    parts = []

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        max_tokens=MAX_TOKENS,
    )

    for chunk in stream:

        if chunk.choices:

            delta = chunk.choices[0].delta.content

            if delta:
                print(delta, end="", flush=True)
                parts.append(delta)

    print()

    answer = "".join(parts)

    logger.success(
        f"Streamed answer ({len(answer)} chars)"
    )

    return answer