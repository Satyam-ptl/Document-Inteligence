"""
Conflict detection between retrieved passages (Phase 7).

The project's "reliability" requirement includes flagging when a document
corpus contains **contradictory evidence** for the same question — e.g.
two different reports stating different revenue figures for the same
quarter. A real NLI (natural-language-inference) contradiction-detection
model would be the principled way to do this, but no such model is
installable in this sandbox: it would need the same `FlagEmbedding`/
`torch` stack that's already blocked for the embedder/reranker (no
`huggingface.co` access — see PROJECT_STATE.md), and even with network
access, adding a whole second heavyweight model just for this one Phase 7
feature was judged not worth it versus a lexical heuristic that catches
the concrete, common case this project's demo question set actually needs
("what was the revenue" when two documents disagree).

**What this catches**: two passages that mention what looks like the same
underlying fact — a number with a similar surrounding sentence — but with
meaningfully different numeric values. Two-step heuristic:
  1. Extract every "numeric fact" from every passage: a number (money,
     percentage, or plain count/magnitude, handling common suffixes like
     "million"/"M"/"bn"/"%") plus the *other* content words in the same
     sentence, as a lightweight stand-in for "what is this number about."
  2. Compare every pair of facts from two *different* passages: if their
     surrounding-word sets overlap enough (Jaccard similarity — the same
     "probably talking about the same thing" heuristic) but the numeric
     values differ by more than a small relative tolerance, flag it as a
     conflict.

**What this does NOT catch** (known, accepted limitations — flag for
review, not a completeness guarantee): purely qualitative disagreements
("Report A says the launch succeeded, Report B says it was delayed"),
contradictions that don't involve a number, or two numbers about
genuinely different things that happen to share surrounding vocabulary
(a false positive the Jaccard threshold is tuned to keep rare, not
impossible). This is a recall-oriented safety net for the numeric case,
not a general-purpose fact-checker.

Pure regex/string logic, zero external dependency — same rigor as
`citation.py`: run directly against real scenarios this session (see
PROJECT_STATE.md's Phase 7 verification log).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.generation.citation import split_sentences

# Numbers with an optional leading currency symbol, thousands separators, a
# decimal part, and an optional magnitude/percent suffix. Deliberately
# permissive (matches "5", "$5.2", "38%", "1,234", "5.2 million", "6.1bn")
# over strict, since a missed number is a missed conflict-detection
# opportunity but an over-matched one is just an extra (usually harmless)
# candidate fact to compare. The leading negative lookbehind excludes a
# digit immediately glued to a preceding letter/digit (e.g. the "3" in
# "Q3", or an alphanumeric ID) — without it, "Q3 2025" spuriously yields a
# bogus standalone "3" fact that can then be compared against unrelated
# numbers elsewhere and produce a false-positive conflict (caught by this
# session's Phase 7 verification run, not a theoretical concern).
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<sign>-)?\$?(?P<num>\d[\d,]*\.?\d*)\s*(?P<suffix>%|million|billion|thousand|bn|m\b|k\b)?",
    re.IGNORECASE,
)

_MAGNITUDE_MULTIPLIERS = {
    "million": 1_000_000.0,
    "m": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "thousand": 1_000.0,
    "k": 1_000.0,
}

# Small, deliberately conservative stopword list — just enough to keep
# near-universal filler words from inflating Jaccard similarity between
# two sentences that aren't really about the same thing. Not a linguistic
# stopword list; tuned for this one heuristic's purpose only.
_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "was", "were", "is", "are", "that", "this",
    "with", "by", "as", "at", "from", "its", "their", "it", "which", "who", "has", "have", "had", "be", "been",
    "will", "would", "could", "should", "also", "than", "over", "up", "down", "into", "about", "such", "not",
    "we", "our", "they", "these", "those", "there", "then", "so", "but", "if", "each", "per",
}

_WORD_RE = re.compile(r"[a-zA-Z]{2,}")

# Two facts sharing at least this fraction of context words (Jaccard
# similarity) are treated as "probably about the same underlying thing" —
# tuned conservatively (favoring precision over recall) since a false
# conflict flag actively misleads a reader, while a missed one just means
# no flag was raised (same failure mode as not having this feature at all).
_SIMILARITY_THRESHOLD = 0.5

# Two facts about the same thing whose values differ by more than this
# fraction of the larger value are flagged as conflicting. 5% comfortably
# exceeds normal rounding/reporting differences (e.g. "$5.0M" vs "$5.02M")
# while still catching a materially different figure (e.g. "$5M" vs "$6M").
_RELATIVE_TOLERANCE = 0.05

# Caps total work/output on a pathologically large passage set — same
# defensive-truncation pattern used throughout this codebase (e.g.
# `hybrid.py`'s `RETRIEVAL_RERANK_POOL`). In practice `len(passages)` is
# already bounded by `RETRIEVAL_TOP_K_FINAL` (default 8), so this rarely
# matters; it exists so a future caller passing a much larger passage list
# fails safe (fewer reported conflicts) rather than slow or unbounded.
_MAX_FACTS = 200
_MAX_CONFLICTS = 20


@dataclass
class NumericFact:
    passage_index: int  # matches ContextPassage.index / the [n] citation convention
    raw_text: str  # the number exactly as it appeared, e.g. "$5.2 million"
    value: float  # normalized numeric value (magnitude suffix applied; percent is NOT rescaled)
    unit_type: str  # "percent" | "magnitude" | "plain" — only same-unit-type facts are ever compared
    context_tokens: frozenset[str]
    sentence: str


@dataclass
class ConflictPair:
    passage_index_a: int
    passage_index_b: int
    text_a: str
    text_b: str
    sentence_a: str
    sentence_b: str
    shared_context: list[str]  # sorted overlapping context words, for explainability in the API/UI


@dataclass
class ConflictDetectionResult:
    has_conflict: bool
    conflicts: list[ConflictPair]


def _normalize_value(raw_num: str, suffix: str | None, sign: str | None) -> tuple[float, str]:
    value = float(raw_num.replace(",", ""))
    if sign:
        value = -value
    if suffix is None:
        return value, "plain"
    suffix_lower = suffix.lower()
    if suffix_lower == "%":
        return value, "percent"
    multiplier = _MAGNITUDE_MULTIPLIERS.get(suffix_lower)
    if multiplier is not None:
        return value * multiplier, "magnitude"
    return value, "plain"  # defensive: shouldn't happen given the regex's own suffix alternatives


def _context_tokens(sentence: str, matched_span: tuple[int, int]) -> frozenset[str]:
    start, end = matched_span
    remainder = sentence[:start] + " " + sentence[end:]
    return frozenset(
        w.lower() for w in _WORD_RE.findall(remainder) if w.lower() not in _STOPWORDS
    )


def _extract_facts(passages: list[Any]) -> list[NumericFact]:
    facts: list[NumericFact] = []
    for passage in passages:
        for sentence in split_sentences(passage.text):
            for match in _NUMBER_RE.finditer(sentence):
                if len(facts) >= _MAX_FACTS:
                    return facts
                value, unit_type = _normalize_value(match.group("num"), match.group("suffix"), match.group("sign"))
                facts.append(
                    NumericFact(
                        passage_index=passage.index,
                        raw_text=match.group(0).strip(),
                        value=value,
                        unit_type=unit_type,
                        context_tokens=_context_tokens(sentence, match.span()),
                        sentence=sentence,
                    )
                )
    return facts


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / len(a | b)


def detect_conflicts(
    passages: list[Any],
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    relative_tolerance: float = _RELATIVE_TOLERANCE,
) -> ConflictDetectionResult:
    """Flag pairs of passages that appear to state different numeric
    values for what looks like the same underlying fact.

    `passages` is a list of `app/generation/prompt.py`'s `ContextPassage`
    (or anything with the same `.index`/`.text` attributes) — the same
    numbered passages actually offered to the LLM, so a flagged conflict's
    `passage_index_a`/`_b` line up directly with the `[n]` citation
    markers in the generated answer.
    """
    facts = _extract_facts(passages)
    conflicts: list[ConflictPair] = []
    seen_pairs: set[tuple[int, int, str, str]] = set()

    for i, fact_a in enumerate(facts):
        if len(conflicts) >= _MAX_CONFLICTS:
            break
        for fact_b in facts[i + 1 :]:
            if fact_a.passage_index == fact_b.passage_index:
                continue  # only cross-passage disagreement counts as a "conflict" between sources
            if fact_a.unit_type != fact_b.unit_type:
                continue  # e.g. never compare a percentage against a dollar amount
            similarity = _jaccard(fact_a.context_tokens, fact_b.context_tokens)
            if similarity < similarity_threshold:
                continue
            larger = max(abs(fact_a.value), abs(fact_b.value))
            if larger == 0:
                continue
            relative_diff = abs(fact_a.value - fact_b.value) / larger
            if relative_diff <= relative_tolerance:
                continue

            a, b = (fact_a, fact_b) if fact_a.passage_index <= fact_b.passage_index else (fact_b, fact_a)
            dedupe_key = (a.passage_index, b.passage_index, a.raw_text, b.raw_text)
            if dedupe_key in seen_pairs:
                continue
            seen_pairs.add(dedupe_key)

            conflicts.append(
                ConflictPair(
                    passage_index_a=a.passage_index,
                    passage_index_b=b.passage_index,
                    text_a=a.raw_text,
                    text_b=b.raw_text,
                    sentence_a=a.sentence,
                    sentence_b=b.sentence,
                    shared_context=sorted(a.context_tokens & b.context_tokens),
                )
            )
            if len(conflicts) >= _MAX_CONFLICTS:
                break

    return ConflictDetectionResult(has_conflict=bool(conflicts), conflicts=conflicts)
