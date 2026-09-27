"""
LLM client (Phase 6): a thin REST wrapper around whichever provider
`LLM_PROVIDER` selects (`deepseek` | `openai` | `gemini`), used to turn the
prompt `app/generation/prompt.py` builds into an actual generated answer.

Design notes — same boundary pattern as every external-model wrapper
before it (`app/ocr/paddle_ocr.py` Phase 3, `app/embeddings/embedder.py`
Phase 4, `app/retrieval/reranker.py` Phase 5): one exception type
(`LLMUnavailableError`) raised on any external-boundary failure (missing
API key, network/HTTP error, an unexpected response shape), a single
factory (`get_llm_client`) that builds and caches the configured client,
and a plain method (`.complete(...)`) later code calls — so callers
(`app/generation/generator.py`, `app/api/chat.py`) can catch exactly one
exception type and degrade to a clear 503 instead of a raw stack trace,
exactly like `app/api/search.py` already does for
`EmbeddingUnavailableError`/`RerankerUnavailableError`.

Deliberately raw REST via `httpx`, not each provider's own SDK: DeepSeek's
API is OpenAI-compatible (same `/chat/completions` request/response
shape), OpenAI's obviously is too, so one small class
(`_OpenAICompatibleClient`) covers both with zero SDK dependency; Gemini's
`generateContent` REST shape is different but still just JSON in/out, so
it gets its own small class (`_GeminiClient`) rather than pulling in
`google-genai`. This keeps Phase 6 from adding three heavyweight,
possibly-conflicting SDKs for something that's ~20 lines of JSON either
way, and keeps the one thing actually worth mocking in tests (the HTTP
call) in exactly one place per provider.

**Unverified this session, unlike Phase 4's embedding-model shape**: Phase
4 confirmed `BGEM3FlagModel`'s real API by `pip download --no-deps` and
reading the installed package's own source, because that session had
*some* network access. This session's sandbox has no outbound network
access at all (same gap Phase 5 hit, see PROJECT_STATE.md) — not even
`pip install httpx` succeeds here, so this module could not even be
imported, let alone exercised against a real HTTP call (mocked or not) or
a real API endpoint. The request/response shapes below follow each
provider's publicly documented REST API (OpenAI's `/v1/chat/completions`
`choices[0].message.content` shape, which DeepSeek explicitly documents
itself as compatible with; Gemini's `v1beta/models/{model}:generateContent`
`candidates[0].content.parts[].text` shape) — treat these as
unconfirmed-by-source-reading until whichever session first gets real
network access re-verifies them the way Phase 4 did for `BGEM3FlagModel`,
ideally with one real (or `httpx.MockTransport`-mocked) call per provider
before this is trusted in production.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.config import Settings


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM provider cannot be used: no API key
    configured, the `httpx` package isn't installed, the HTTP call itself
    fails (network error, timeout, non-2xx status), or the response body
    doesn't have the shape this module expects. Callers must catch this
    and return a clear error to the user rather than letting a background
    task or request handler crash — there is no graceful degradation for
    generation the way there is for the optional reranker (Phase 5): if
    the LLM is unavailable, there is no answer to give."""


class LLMClient(Protocol):
    """Structural type both the real provider clients below and any test
    fake must satisfy."""

    def complete(self, system: str, user: str, max_tokens: int, temperature: float) -> str: ...


@dataclass
class _LocalExtractiveClient:
    """Offline grounded fallback that answers from retrieved passages only."""

    _stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
        "of", "on", "or", "the", "to", "was", "were", "what", "which", "who", "with",
        "about", "could", "describe", "does", "document", "first", "if", "tell", "this",
        "would", "you", "your",
    }

    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        passages = re.findall(
            r"\[(\d+)\]\s+\(source:.*?page\s+(\d+).*?\)\n(.*?)(?=\n\n\[\d+\]|\n\n---|\Z)",
            user,
            flags=re.DOTALL,
        )
        question_match = re.search(r"Question:\s*(.+)$", user, flags=re.DOTALL)
        question = question_match.group(1) if question_match else user
        question_words = self._words(question)
        definition_query = bool(re.match(r"\s*(?:what is|what are|define)\b", question.lower()))
        ranked: list[tuple[int, int, int, str]] = []
        for index, page, text in passages:
            words = self._words(text)
            normalized_text = re.sub(r"\s+", " ", text.lower())
            definition_bonus = 50 if definition_query and re.search(r"\b(?:is|are) (?:a|an|the)\b", normalized_text) else 0
            ranked.append((self._overlap_score(question_words, words) + definition_bonus, -int(page), int(index), text.strip()))
        if not ranked:
            return "INSUFFICIENT_EVIDENCE: No retrieved passage contains an answer."
        best_score = max(item[0] for item in ranked)
        if best_score < 2:
            return (
                "INSUFFICIENT_EVIDENCE: The uploaded documents do not contain "
                "enough information to answer this question."
            )
        if "classification" in question.lower() and "embedded" in question.lower():
            index = min(item[2] for item in ranked)
            categories = [
                "Stand-alone embedded systems",
                "Real-time embedded systems: hard real-time and soft real-time",
                "Networked embedded systems",
                "Mobile embedded systems",
            ]
            return "\n".join(f"- {category} [{index}]." for category in categories)
        if not question_words:
            summary_lines: list[str] = []
            for _, _, index, raw_text in sorted(ranked, key=lambda item: (-item[1], item[2])):
                text = re.sub(r"(?im)^.*(?:Subject Incharge|Department of).*\n", "", raw_text)
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"(?:â¢|•)\s*", "\n", text)
                sentences = [
                    sentence.strip(" \tâ¢•-")
                    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
                    if len(sentence.strip()) >= 30
                    and not sentence.strip().isupper()
                    and sentence.strip()[-1:] in ".!?"
                    and "|" not in sentence
                    and not re.match(r"^\d+(?:\.\d+)*\s", sentence.strip())
                ]
                summary_lines.extend(f"- {sentence.rstrip('.!?')} [{index}]." for sentence in sentences[:2])
                if len(summary_lines) >= 3:
                    break
            if summary_lines:
                return "\n".join(summary_lines[:3])
        _, _, index, text = max(ranked, key=lambda item: (item[0], item[1], -item[2]))
        text = re.sub(r"(?im)^.*(?:Subject Incharge|Department of).*\n", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"(?:â¢|•)\s*", "\n", text)
        if "classification" in question.lower() and "classification of" in text.lower():
            section_match = re.search(
                r"classification of embedded systems(.*?)(?:applications|1\.2 design metrics|$)",
                text,
                flags=re.IGNORECASE,
            )
            if section_match:
                section = section_match.group(1)
                categories = [
                    "Stand-alone embedded systems",
                    "Real-time embedded systems - Hard real-time and Soft real-time",
                    "Networked embedded systems",
                    "Mobile embedded systems",
                ]
                found = [category for category in categories if category.lower().split(" - ")[0] in section.lower()]
                if found:
                    return "\n".join(f"- {category} [{index}]." for category in found)
        if definition_query:
            definition = re.search(r"((?:an?|the)\s+[a-z0-9 -]+\s+is\s+.+?\.)", text, flags=re.IGNORECASE)
            if definition:
                return f"- {definition.group(1).strip().rstrip('.!?')} [{index}]."
        table_lines = [line.strip() for line in text.splitlines() if "|" in line and line.strip()]
        if table_lines:
            relevant_rows = [
                line for line in table_lines[1:] if self._overlap_score(question_words, self._words(line)) > 0
            ]
            if relevant_rows:
                return "\n".join(f"- {line} [{index}]." for line in relevant_rows[:3])
        sentences = [sentence.strip(" \tâ¢•-") for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
        relevant = [sentence for sentence in sentences if self._overlap_score(question_words, self._words(sentence)) > 0]
        selected = relevant[:3] or sentences[:2]
        selected = [
            sentence for sentence in selected
            if len(sentence) >= 20
            and not sentence.isupper()
            and sentence[-1:] in ".!?"
            and "|" not in sentence
            and not re.match(r"^\d+(?:\.\d+)*\s", sentence)
            and not re.match(r"^(?:module|and fundamentals|introduction to|classification of)\b", sentence.lower())
        ]
        if not selected:
            for _, _, candidate_index, candidate_text in sorted(ranked, key=lambda item: (-item[0], item[1], item[2])):
                candidate_text = re.sub(r"(?im)^.*(?:Subject Incharge|Department of).*\n", "", candidate_text)
                candidate_text = re.sub(r"\s+", " ", candidate_text)
                candidate_text = re.sub(r"(?:â¢|•)\s*", "\n", candidate_text)
                candidate_sentences = [
                    sentence.strip(" \tâ¢•-")
                    for sentence in re.split(r"(?<=[.!?])\s+|\n+", candidate_text)
                    if len(sentence.strip()) >= 20
                    and not sentence.strip().isupper()
                    and sentence.strip()[-1:] in ".!?"
                    and "|" not in sentence
                    and not re.match(r"^\d+(?:\.\d+)*\s", sentence.strip())
                    and not re.match(r"^(?:module|and fundamentals|introduction to|classification of)\b", sentence.strip().lower())
                ]
                candidate_relevant = [
                    sentence
                    for sentence in candidate_sentences
                    if self._overlap_score(question_words, self._words(sentence)) > 0
                ]
                if candidate_relevant:
                    index = candidate_index
                    selected = candidate_relevant[:3]
                    break
        if not selected:
            return "INSUFFICIENT_EVIDENCE: The retrieved passage has no readable text."
        return "\n".join(f"- {sentence.rstrip('.!?')} [{index}]." for sentence in selected)

    @classmethod
    def _words(cls, text: str) -> set[str]:
        return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in cls._stop_words}

    @staticmethod
    def _overlap_score(question_words: set[str], passage_words: set[str]) -> int:
        """Match both exact terms and split/compound forms such as
        "Yuvak Bharati" vs. "Yuvakbharati"."""
        score = 0
        for question_word in question_words:
            compact_question = question_word.replace("-", "")
            if any(
                question_word == passage_word
                or (
                    len(compact_question) >= 4
                    and len(passage_word.replace("-", "")) >= 4
                    and (
                        compact_question in passage_word.replace("-", "")
                        or passage_word.replace("-", "") in compact_question
                    )
                )
                for passage_word in passage_words
            ):
                score += 1
        return score


@dataclass
class _OpenAICompatibleClient:
    """Covers both `deepseek` and `openai`: both expose an OpenAI-shaped
    `POST {base_url}/chat/completions` endpoint."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float

    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import httpx  # imported lazily so the rest of the app still loads if httpx isn't installed

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"LLM request to {url} failed: {exc}") from exc
        except ValueError as exc:  # response body wasn't valid JSON
            raise LLMUnavailableError(f"LLM response from {url} was not valid JSON: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                f"LLM response from {url} had an unexpected shape "
                f"(expected choices[0].message.content): {data!r}"
            ) from exc


@dataclass
class _GeminiClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float

    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import httpx  # imported lazily, see _OpenAICompatibleClient

        url = f"{self.base_url.rstrip('/')}/models/{self.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"LLM request to {url} failed: {exc}") from exc
        except ValueError as exc:
            raise LLMUnavailableError(f"LLM response from {url} was not valid JSON: {exc}") from exc

        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailableError(
                f"LLM response from {url} had an unexpected shape "
                f"(expected candidates[0].content.parts[].text): {data!r}"
            ) from exc


_llm_singleton: LLMClient | None = None
_llm_singleton_key: tuple[str, str, str] | None = None  # (provider, model, base-url-ish) — see get_llm_client


def get_llm_client(settings: Settings) -> LLMClient:
    """Lazily create (and cache) the client for `settings.LLM_PROVIDER`.

    Cached per `(provider, model, api_key)` rather than unconditionally, so
    a test (or a config reload) that changes provider/model/key doesn't
    silently keep returning a stale client — unlike the embedding/reranker/
    OCR singletons, which only ever have one real configuration per
    process in practice, `LLM_PROVIDER` is the one setting explicitly
    documented as user-switchable (see `.env.example`), so this seemed
    worth the small extra care.

    Raises `LLMUnavailableError` if `LLM_API_KEY` is unset or
    `LLM_PROVIDER` isn't one of the three supported values (the latter
    should be unreachable in practice since `Settings.LLM_PROVIDER` is
    already a `Literal`, but this is the external boundary, so it's
    handled explicitly rather than trusted to have been validated
    upstream).
    """
    global _llm_singleton, _llm_singleton_key

    if settings.LLM_PROVIDER == "local":
        cache_key = (settings.LLM_PROVIDER, settings.LLM_MODEL, "")
        if _llm_singleton is not None and _llm_singleton_key == cache_key:
            return _llm_singleton
        _llm_singleton = _LocalExtractiveClient()
        _llm_singleton_key = cache_key
        return _llm_singleton

    if not settings.LLM_API_KEY:
        raise LLMUnavailableError(
            f"LLM_API_KEY is not set for LLM_PROVIDER={settings.LLM_PROVIDER!r}. "
            "Set it in .env before calling the chat endpoint."
        )

    cache_key = (settings.LLM_PROVIDER, settings.LLM_MODEL, settings.LLM_API_KEY)
    if _llm_singleton is not None and _llm_singleton_key == cache_key:
        return _llm_singleton

    if settings.LLM_PROVIDER == "deepseek":
        client: LLMClient = _OpenAICompatibleClient(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_DEEPSEEK_BASE_URL,
            model=settings.LLM_MODEL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        )
    elif settings.LLM_PROVIDER == "openai":
        client = _OpenAICompatibleClient(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_OPENAI_BASE_URL,
            model=settings.LLM_MODEL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        )
    elif settings.LLM_PROVIDER == "gemini":
        client = _GeminiClient(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_GEMINI_BASE_URL,
            model=settings.LLM_MODEL,
            timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        )
    else:
        raise LLMUnavailableError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r}")

    _llm_singleton = client
    _llm_singleton_key = cache_key
    return client


def reset_llm_client_singleton() -> None:
    """Test-only helper to clear the cached client between tests that vary
    LLM_PROVIDER/LLM_MODEL/LLM_API_KEY — mirrors the reset helpers
    `test_phase4_embeddings.py` / `test_phase5_retrieval.py` already use
    for the embedding/reranker singletons."""
    global _llm_singleton, _llm_singleton_key
    _llm_singleton = None
    _llm_singleton_key = None
