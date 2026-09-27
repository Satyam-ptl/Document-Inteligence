"""
Grounded-answer prompt construction (Phase 6).

Builds the system + user prompt sent to the LLM from a list of retrieved
chunks (`app/retrieval/hybrid.py`'s `HybridSearchResult`, or anything else
with the same shape). Two hard rules are baked into the system prompt,
matching the project's "no hallucination, always attribute" requirement:

  1. Answer using ONLY the numbered context passages below — if the answer
     isn't in them, say so explicitly (a fixed `INSUFFICIENT_EVIDENCE:`
     prefix, see `app/generation/citation.py`) rather than guessing from
     outside knowledge.
  2. Every factual sentence must end with one or more bracketed citation
     markers like `[2]` or `[1][3]`, referencing the numbered passage(s) it
     came from. `app/generation/citation.py` parses and validates these
     against the real passage count after the LLM responds — the LLM is
     trusted to follow rule 2, never trusted to have followed it correctly.

Pure string formatting, zero external dependency — real-verified this
session (see PROJECT_STATE.md's Phase 6 verification log), not just
written and assumed to work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ContextPassage:
    """One numbered passage as shown to the LLM. `index` is the 1-based
    number used in the `[n]` marker convention throughout Phase 6 — it is
    NOT the same as `chunk_index` (the chunk's position within its source
    document), which is unrelated and not shown to the LLM at all."""

    index: int
    chunk_id: str
    document_id: str
    source_filename: str
    page_number: int | None
    text: str


SYSTEM_PROMPT = (
    "You are a careful document-analysis assistant. Answer the user's "
    "question using ONLY the numbered context passages provided in the "
    "user message. Never use outside knowledge, and never guess.\n\n"
    "Rules:\n"
    "1. If the passages do not contain enough information to answer the "
    "question, reply with EXACTLY this format: \"INSUFFICIENT_EVIDENCE: \" "
    "followed by one short sentence naming what's missing. Do not "
    "speculate or partially answer from outside knowledge.\n"
    "2. Otherwise, every factual sentence in your answer MUST end with one "
    "or more bracketed citation markers referencing the passage number(s) "
    "it is based on, e.g. \"Revenue grew 12% in 2025 [2].\" or "
    "\"The two figures disagree [1][3].\"\n"
    "3. Never invent or cite a passage number that was not given to you.\n"
    "4. Be concise and answer the question directly; do not restate the "
    "question or describe the passages in general terms."
)


def build_context_block(passages: list[ContextPassage]) -> str:
    """Render the numbered passages exactly as the LLM will see them."""
    if not passages:
        return "(no passages retrieved)"
    lines = []
    for p in passages:
        loc = f"{p.source_filename}, page {p.page_number}" if p.page_number is not None else p.source_filename
        lines.append(f"[{p.index}] (source: {loc})\n{p.text.strip()}")
    return "\n\n".join(lines)


def build_user_prompt(query: str, passages: list[ContextPassage]) -> str:
    return (
        f"Context passages:\n\n{build_context_block(passages)}\n\n"
        f"---\n\nQuestion: {query.strip()}"
    )


def to_context_passages(results: list[Any]) -> list[ContextPassage]:
    """Convert `hybrid_search()` results (or any object exposing the same
    `chunk_id`/`document_id`/`source_filename`/`page_number`/`text`
    attributes — e.g. a test fake) into 1-indexed `ContextPassage` objects,
    preserving the input order. Callers are expected to have already
    ranked/reranked/truncated the list; this function does not reorder or
    filter anything."""
    return [
        ContextPassage(
            index=i + 1,
            chunk_id=r.chunk_id,
            document_id=r.document_id,
            source_filename=r.source_filename,
            page_number=r.page_number,
            text=r.text,
        )
        for i, r in enumerate(results)
    ]
