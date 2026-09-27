"""
Conversation memory / multi-turn follow-up resolution (Phase 8).

Design decisions (raised as open questions in PROJECT_STATE.md's Phase 8
handoff, resolved here):

1. **Storage: a real DB table, not in-memory.** `Conversation` and
   `ConversationTurn` (see `app/database/models.py`) live in the same
   SQLAlchemy DB as every other durable piece of this project's state —
   a chat session should survive a server restart the same way documents
   and chunks do, and per-process in-memory storage would silently lose
   history on every redeploy or multi-worker deployment. A conversation is
   only ever created lazily, the first time a `/api/chat` caller supplies
   a `conversation_id` (see `get_or_create_conversation()`), so callers
   that never opt in keep Phase 6/7's original stateless, single-turn
   contract exactly as-is — no schema change is *required* to use
   `/api/chat` as before.

2. **History window: the last `CONVERSATION_HISTORY_TURNS` turns**
   (default 3, see `app/config.py`), not the whole conversation. Enough
   to resolve "what about last quarter?" against the immediately
   preceding topic, without an old, unrelated turn from far earlier in a
   long-running conversation leaking into the rewrite of a much later
   message. This is a fixed count, not a token budget — simple, and this
   project's chunks/prompts are already short enough that a handful of
   turns is cheap regardless.

3. **Resolution mechanism: one extra small LLM call, not a rule-based
   rewrite.** A hand-rolled rule-based approach (pronoun substitution,
   keyword carry-over) cannot reliably handle the general case: "what
   about last quarter?" is not a pronoun to substitute, it's an entirely
   elided question, and the space of ways a follow-up can depend on prior
   context (elision, comparison, "and the other one?", implicit scope
   narrowing) is exactly the kind of open-ended language understanding an
   LLM call is suited for and a regex is not. This project already pays
   for one LLM round trip per `/api/chat` call (Phase 6's generation
   call); a second, much shorter one (a few dozen tokens of history, a
   short output) is the standard "condense question" step used throughout
   production RAG systems, and is a reasonable incremental cost for
   materially better correctness. `CONVERSATION_REWRITE_ENABLED=False`
   turns it off entirely for deployments that want conversation *tracking*
   (still get conversation_id + history) without the extra call.

   On the very first turn of a conversation (no prior history yet),
   resolution is skipped unconditionally — there is nothing to resolve
   against, so the raw query is used as-is regardless of the setting.
   If the rewrite call itself fails (`LLMUnavailableError`), this
   degrades to the raw query rather than failing the whole request: a
   slightly-worse-resolved follow-up is far better than an outage on a
   feature (Phase 6 generation) that doesn't actually need the rewrite
   step to function.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from sqlalchemy.orm import Session

from app.config import Settings
from app.database.models import Conversation, ConversationTurn
from app.generation.llm_client import LLMClient, LLMUnavailableError

REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's latest chat message into a fully standalone "
    "question, using the conversation history only for context. Rules:\n"
    "1. If the latest message is already a standalone question that does "
    "not depend on anything earlier in the conversation, return it "
    "UNCHANGED.\n"
    "2. Otherwise, rewrite it into one complete, standalone question that "
    "makes sense with no other context, resolving any pronouns, ellipsis, "
    "or implicit references (e.g. \"what about 2024?\" following a "
    "question about 2025 revenue becomes \"What was the revenue in "
    "2024?\").\n"
    "3. Preserve the user's original intent exactly. Do not answer the "
    "question yourself, and do not introduce any fact not implied by the "
    "conversation history.\n"
    "4. Respond with ONLY the rewritten question and nothing else — no "
    "preamble, no quotation marks, no explanation."
)


@dataclass
class ResolvedQuery:
    raw_query: str
    # What was actually sent to retrieval/generation. Equals raw_query.strip()
    # whenever there was no history, rewriting is disabled, the rewrite call
    # failed, or the LLM judged the raw query already standalone.
    resolved_query: str
    rewritten: bool  # True iff resolved_query differs from raw_query.strip()


def build_history_block(turns: list[ConversationTurn]) -> str:
    """Render prior turns oldest-first as plain User/Assistant lines — the
    same flat transcript shape a human would read, and about as cheap as a
    format can be for the rewrite-prompt LLM call to consume."""
    lines = []
    for t in turns:
        lines.append(f"User: {t.raw_query}")
        lines.append(f"Assistant: {t.answer}")
    return "\n".join(lines)


def resolve_followup(
    raw_query: str,
    history_turns: list[ConversationTurn],
    llm_client: LLMClient,
    settings: Settings,
) -> ResolvedQuery:
    """Rewrite `raw_query` into a standalone query if it looks like a
    follow-up, using `history_turns` (oldest-first) as context. See this
    module's docstring for the full reasoning behind each branch below."""
    stripped = raw_query.strip()

    if not history_turns or not settings.CONVERSATION_REWRITE_ENABLED:
        return ResolvedQuery(raw_query=raw_query, resolved_query=stripped, rewritten=False)

    user_prompt = (
        f"Conversation history:\n\n{build_history_block(history_turns)}\n\n"
        f"---\n\nLatest message: {stripped}"
    )
    try:
        rewritten = llm_client.complete(
            system=REWRITE_SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=150,
            temperature=0.0,
        )
    except LLMUnavailableError as exc:
        logger.warning(f"resolve_followup: rewrite call failed, falling back to the raw query — {exc}")
        return ResolvedQuery(raw_query=raw_query, resolved_query=stripped, rewritten=False)

    rewritten = rewritten.strip().strip('"').strip()
    if not rewritten:
        logger.warning("resolve_followup: rewrite call returned an empty string, falling back to the raw query.")
        return ResolvedQuery(raw_query=raw_query, resolved_query=stripped, rewritten=False)

    return ResolvedQuery(raw_query=raw_query, resolved_query=rewritten, rewritten=rewritten != stripped)


def get_or_create_conversation(
    db: Session, conversation_id: str | None, document_id: str | None = None
) -> Conversation:
    """Fetch the conversation for `conversation_id`, or create a new one.

    A caller-supplied id that doesn't exist yet is treated as "start a new
    conversation using this id" rather than a 404 — this keeps
    `/api/chat`'s contract simple (a caller always has *some* id to send
    back on the next turn, whether this was the first call or not) at the
    cost of not being able to tell "brand new" from "typo'd an old id"
    from this function alone; a caller that cares can check whether the
    returned conversation has any turns yet.
    """
    conversation: Conversation | None = None
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        conversation = Conversation(document_id=document_id, **({"id": conversation_id} if conversation_id else {}))
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
    return conversation


def get_recent_turns(db: Session, conversation_id: str, limit: int) -> list[ConversationTurn]:
    """The most recent `limit` turns for a conversation, oldest-first —
    the order both `build_history_block()` and a rendered transcript want
    them in."""
    turns = (
        db.query(ConversationTurn)
        .filter(ConversationTurn.conversation_id == conversation_id)
        .order_by(ConversationTurn.turn_index.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(turns))


def record_turn(
    db: Session, conversation: Conversation, raw_query: str, resolved_query: str, answer: str
) -> ConversationTurn:
    """Append one turn to `conversation`. `turn_index` is derived from a
    fresh count query (not `len(conversation.turns)`) so this is correct
    even if the in-memory `conversation` object's `turns` relationship
    hasn't been refreshed since an earlier turn was written in the same
    process."""
    existing_count = (
        db.query(ConversationTurn).filter(ConversationTurn.conversation_id == conversation.id).count()
    )
    turn = ConversationTurn(
        conversation_id=conversation.id,
        turn_index=existing_count,
        raw_query=raw_query,
        resolved_query=resolved_query,
        answer=answer,
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    return turn
