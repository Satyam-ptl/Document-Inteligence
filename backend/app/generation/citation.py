"""
Citation parsing + validation (Phase 6).

The LLM is instructed (see `app/generation/prompt.py`'s `SYSTEM_PROMPT`) to
end every factual sentence with one or more `[n]` markers referencing a
numbered context passage, or to reply with a fixed `INSUFFICIENT_EVIDENCE:`
prefix if it can't answer from the given passages at all. This module is
the other half of that contract: after the LLM responds, it extracts every
citation marker actually used and checks each one against how many
passages really existed — this is the project's actual "citation
validation" requirement, not just parsing. The LLM being *told* the rules
does not mean it followed them; nothing here assumes it did.

Two distinct failure modes this catches, both real and both worth
distinguishing downstream (Phase 7's reliability/confidence scoring will
want to treat them differently):
  - an **invalid citation**: the answer cites `[7]` but only 4 passages
    were ever given to the LLM — an outright hallucinated reference.
  - an **uncited sentence**: a factual-looking sentence with no `[n]`
    marker at all — the LLM followed rule 1 (used the context) but broke
    rule 2 (attribute it), which is just as much a problem for a system
    whose whole point is source attribution.

Pure regex/string logic, zero external dependency — real-verified this
session (see PROJECT_STATE.md's Phase 6 verification log): every scenario
in this module's docstrings was actually run against this code with
`python3`, output inspected by hand, not just written and assumed correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_INSUFFICIENT_EVIDENCE_PREFIX = "INSUFFICIENT_EVIDENCE:"

# Splits after terminal punctuation (with any immediately-following [n]
# markers kept glued to the sentence they close), on whitespace. This is
# deliberately a blunt heuristic, not a linguistic sentence tokenizer — it
# only needs to be good enough to flag "does this sentence have a citation
# marker," not to be correct on every edge case of English punctuation
# (e.g. "Fig. 2" would be mis-split; acceptable for this project's scope).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:\[\d+\])*\s+")


@dataclass
class CitationValidationResult:
    is_insufficient_evidence: bool
    cited_indices: list[int]  # every unique [n] found in the answer, in first-seen order
    valid_indices: list[int]  # cited indices that correspond to a real passage (1..num_passages)
    invalid_indices: list[int]  # cited indices with no matching passage — a hallucinated citation
    uncited_sentence_count: int  # factual sentences with no citation marker at all
    total_sentence_count: int
    fully_cited: bool  # True iff every sentence has >=1 valid citation and there are zero invalid ones


def is_insufficient_evidence_answer(answer: str) -> bool:
    return answer.strip().upper().startswith(_INSUFFICIENT_EVIDENCE_PREFIX)


def extract_citation_indices(answer: str) -> list[int]:
    """All unique [n] markers in the answer, in first-seen order.

    >>> extract_citation_indices("A [1]. B [2][1]. C.")
    [1, 2]
    """
    seen: list[int] = []
    for match in _CITATION_MARKER_RE.finditer(answer):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in text.strip().splitlines():
        sentences.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(line.strip()) if s.strip())
    return sentences


def validate_citations(answer: str, num_passages: int) -> CitationValidationResult:
    """Check an LLM answer's citations against how many context passages it
    was actually given (`num_passages` — the length of the
    `app/generation/prompt.py` `ContextPassage` list built for this
    request). `num_passages` is the ground truth for which `[n]` values
    are legitimate; anything outside `1..num_passages` is flagged as
    invalid no matter how confidently the LLM used it.

    An `INSUFFICIENT_EVIDENCE:` answer is treated as `fully_cited=True` —
    there is nothing to cite, and an honest "I don't know" is the correct,
    non-hallucinating behavior this whole module exists to encourage, not
    a citation failure.
    """
    if is_insufficient_evidence_answer(answer):
        return CitationValidationResult(
            is_insufficient_evidence=True,
            cited_indices=[],
            valid_indices=[],
            invalid_indices=[],
            uncited_sentence_count=0,
            total_sentence_count=0,
            fully_cited=True,
        )

    cited = extract_citation_indices(answer)
    valid = [n for n in cited if 1 <= n <= num_passages]
    invalid = [n for n in cited if n not in valid]

    sentences = split_sentences(answer)
    uncited = sum(1 for s in sentences if not _CITATION_MARKER_RE.search(s))

    return CitationValidationResult(
        is_insufficient_evidence=False,
        cited_indices=cited,
        valid_indices=valid,
        invalid_indices=invalid,
        uncited_sentence_count=uncited,
        total_sentence_count=len(sentences),
        fully_cited=(uncited == 0 and not invalid and len(sentences) > 0),
    )
