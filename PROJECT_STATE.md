# PROJECT_STATE.md — Document Intelligence Platform

**Purpose of this file:** this is the single source of truth for resuming
this build in a brand-new chat session (e.g. if your token budget/session
runs out mid-project). Do not delete or heavily edit this file yourself —
each phase's implementation should append to it, not rewrite it.

## HOW TO RESUME IN A NEW SESSION

1. Start a new conversation with Claude.
2. Upload (or paste the contents of) this `PROJECT_STATE.md` file, and
   mention you also have the full project zip if you want Claude to
   read/verify actual files rather than trust this summary.
3. Say: **"Continue the Document Intelligence Platform project from
   PROJECT_STATE.md. Last completed: Phase 12 (repo hygiene — git repo
   initialized and gitignore-verified, README status banner refreshed,
   LICENSE added, CI workflow added). All further KNOWN GAPS items are
   blocked on external resources this sandbox doesn't have (a Docker
   daemon, enough disk for torch/FlagEmbedding, a real LLM API key, open
   network to huggingface.co) — start with whichever of those the new
   environment actually has (see KNOWN GAPS' numbered list, in priority
   order)."** (this is the current handoff prompt as of this file's last
   edit — see `## CURRENT STATUS` above for anything more recent).
4. Claude should re-read this file, treat "Verified Decisions" as locked
   (don't re-litigate them), and continue the incremental phase-by-phase
   process — one phase at a time, with install commands, run instructions,
   and test steps after each phase, exactly as this project started.

---

## CURRENT STATUS

**Last completed: Phase 12 — repo hygiene / prep for a real git host.
REAL-VERIFIED this session: `git init` was actually run in this sandbox
(no repo existed before), `git add -A` staged exactly 95 files, and
`git status --short` / `git check-ignore -v` were used to directly confirm
`.gitignore` behaves exactly as intended (see the Phase 12 summary below
for the specific checks run).**
**Next phase: there isn't a well-defined "Phase 13" — every remaining item
in KNOWN GAPS below is blocked on a resource this sandbox has never had
(Docker daemon, disk for torch/FlagEmbedding, open network to
huggingface.co, a real LLM API key). The next session should pick up
whichever of those its actual environment provides — see KNOWN GAPS' items
2/5/8/9/10/11 for exactly what to do with each one once available. This
repo is otherwise feature-complete for everything buildable without those
resources.**

### Phase 12 design & implementation summary

- **Git repo initialized for real** (`git init`, plus a placeholder
  `user.name`/`user.email` for this session's commit — a future session
  pushing to an actual remote should reconfigure these). No `.git`
  directory existed anywhere in the project before this phase; every prior
  phase's "verification" was necessarily file-existence/content checks,
  never a real commit history.
- **`.gitignore` verified, not just written.** `git add -A` followed by
  `git status --short` confirmed: only `data/{uploads,processed,samples}/
  .gitkeep` are staged from the `data/` trees (all 25 real PNG files
  Phase 3's OCR rasterizer produced against the sample documents in this
  zip are correctly excluded); `git check-ignore -v
  backend/data/uploads/somefile.pdf` explicitly confirmed the Phase-11-
  fixed unanchored `**/data/uploads/*` pattern matches inside `backend/`
  too, not just the repo root; no `node_modules/`, `dist/`, `.venv/`, or
  `.env` appeared in the staged list. This is a genuine behavioral
  verification of the exact bug Phase 11 found and fixed, not a re-read of
  the file.
- **README.md's status banner was several phases stale** (it still said
  "Phases 1-4 verified" and "no outbound network access at all" as of
  Phase 4 — never updated across Phases 5-11). Rewrote it to accurately
  reflect current status: backend/frontend both real-verified via their
  test suites, and the three genuinely-still-open external-resource gaps
  (OCR/embedding model weights, LLM API key, Docker daemon) named
  explicitly rather than implied. The rest of the README (architecture
  diagram, section 7's Docker caveat, section 12's License mention) was
  read and found already accurate — only the top banner needed fixing.
- **`LICENSE` added (MIT).** `README.md`'s own section 12 already stated
  "MIT (adjust as needed for your submission)" from an earlier phase, so
  adding the actual `LICENSE` file makes the repo consistent with what it
  already claimed rather than introducing a new licensing decision this
  session invented — flagged here explicitly in case a future session (or
  the person resuming this project) wants a different license for their
  actual submission/use case.
- **`.github/workflows/ci.yml` added** (KNOWN GAPS item 15, folded into
  this phase since it's the same "prep for a real git host" concern): two
  jobs, `backend` (`pip install` + `pytest tests/ -q`) and `frontend`
  (`npm install` + `tsc -b` + `npm run build` + `vitest run`), mirroring
  exactly the commands already manually verified in the Phase 10/11
  verification logs. Validated only as far as this sandbox allows: parsed
  with `pyyaml` to confirm it's syntactically valid YAML with the two
  expected job names. It has NEVER actually been run by GitHub Actions (no
  repo host access here) — treat its first real push to an actual GitHub
  repo as this workflow's true first test.
- **Not done, explicitly out of scope:** no actual `git commit` was made
  permanent as part of the delivered zip (a `.git` directory with a real
  commit only matters once pushed somewhere; shipping one inside a zip a
  person will re-upload provides little value and needlessly bloats the
  download) — the git init/add/status verification above was to prove the
  gitignore and file set are correct, not to hand over repo history. No
  `CONTRIBUTING.md`/`CODE_OF_CONDUCT.md`/issue templates were added — those
  are genuinely optional polish with no functional bearing on whether the
  project runs, and weren't asked for.

### Phase 11 design & implementation summary

No Docker daemon → nothing here could be run end-to-end. Instead, every
file `docker compose up` would touch was read line-by-line against what
the rest of the (real, tested) codebase actually requires, which surfaced
four concrete bugs — each one is a "this would have failed/silently
misbehaved on first real run" finding, not a style nit:

1. **`docker-compose.yml`'s `backend` service never set `QDRANT_MODE`.**
   `app/retrieval/qdrant_store.py` branches on `QDRANT_MODE` — `"local"`
   (the default baked into `app/config.py`, and what `.env`/`.env.example`
   both set for dev) opens an embedded on-disk Qdrant at
   `QDRANT_LOCAL_PATH`; only `"server"` actually connects to a URL. The
   compose file overrode `QDRANT_URL` to point at the `qdrant` service but
   never overrode `QDRANT_MODE`, so the backend container would have
   silently used its own embedded local DB instead — the `qdrant` service
   would start, sit there unused, and every vector ever indexed would live
   only inside the backend container's ephemeral filesystem instead of the
   `qdrant_data` volume. Fixed: added `QDRANT_MODE: server` next to the
   existing `QDRANT_URL` override.
2. **`backend/requirements.txt` had no Postgres driver.** `.env.example`'s
   own Phase-11 comment already says to switch `DATABASE_URL` to
   `postgresql+psycopg2://...` for docker-compose, and the compose file
   does exactly that — but SQLAlchemy's `psycopg2` dialect needs the
   `psycopg2` (or `psycopg2-binary`) package installed, which was never in
   `requirements.txt` (only `sqlalchemy` itself, which is DB-agnostic).
   The backend container would have crashed on startup with
   `ModuleNotFoundError: No module named 'psycopg2'` the moment it tried to
   connect. Fixed: added `psycopg2-binary==2.9.13` (verified against live
   PyPI this session — latest available — and actually
   `pip install`ed + `import psycopg2`'d successfully in this sandbox's
   venv, confirming the binary wheel installs cleanly with no system
   Postgres headers needed).
3. **`backend/Dockerfile` was missing OS shared libraries `opencv-python-headless`/`paddlepaddle` need.**
   Its only comment on this (a leftover from before Phase 3 was
   implemented) said these libs would be needed "once PaddleOCR-VL-1.6 is
   added" — but Phase 3 added it several phases ago, and the Dockerfile
   was never updated to match. Without `libgomp1` (OpenMP, needed for
   paddlepaddle's CPU inference) and `libglib2.0-0`/`libsm6`/`libxext6`
   (still dynamically linked by opencv-python-headless's image codecs
   despite "headless" only meaning "no GUI/X11 bindings"), `import cv2` or
   `import paddle` would fail at container startup with an `ImportError` —
   after a `pip install` that appears to succeed, since these are runtime
   shared-library link failures, not install-time ones. Fixed: added an
   `apt-get install` step for all four packages, and rewrote the stale
   comment.
4. **The production frontend container had no way to reach the backend at all.**
   `services/api.ts` reads `import.meta.env.VITE_API_BASE_URL` and falls
   back to `""` (a relative `/api/...` path) when it's unset — which only
   works in dev because `vite.config.ts` proxies `/api` to
   `localhost:8000` itself. The production build (`frontend/Dockerfile`'s
   `RUN npm run build`) bakes whatever `VITE_API_BASE_URL` was at *build*
   time directly into the compiled JS (Vite inlines `VITE_`-prefixed env
   vars; nothing about it is configurable at container-run time), and
   neither `frontend/Dockerfile` nor `docker-compose.yml` ever set it. The
   built nginx image also has no reverse-proxy config for `/api` — it just
   serves the static `dist/` output nginx's default config produces. Net
   effect: every fetch from the browser would have gone to
   `http://localhost:5173/api/...` (nginx, no `/api` route configured) and
   404'd, even with every other service healthy. Fixed with the simpler of
   two options (the other being writing an nginx reverse-proxy config from
   scratch): a Docker `ARG`/`ENV` in `frontend/Dockerfile` defaulting
   `VITE_API_BASE_URL` to `http://localhost:8000` — correct here because
   the *browser* (not another container) is what actually calls this URL,
   and the browser is outside the Compose network entirely, so it needs
   the host-published port (`8000:8000`), not the internal `backend`
   service name. `docker-compose.yml`'s `frontend` service was updated to
   pass this explicitly via `build.args` rather than relying silently on
   the Dockerfile default, so the wiring is visible in one place.
5. **`.gitignore` gap found while fixing the above (folded into this
   phase since it's the same docker-data surface, ahead of item 3's
   original "check before Phase 12" scheduling):** the original patterns
   (`data/uploads/*`, `data/processed/*`) were anchored to the repo root
   (a pattern containing a `/` is anchored to the `.gitignore`'s own
   directory in git's matching rules), so they only ever matched
   `document-intelligence/data/...` — never `backend/data/...`, which is
   the separate on-disk tree actually used for local/non-docker runs (this
   is where `pages/*.png` from every OCR run in the uploaded zip actually
   lives). `data/qdrant/` was never ignored at all, at either path. Fixed:
   switched to unanchored `**/data/...` patterns covering both trees, and
   added `**/data/qdrant/`. (`*.db` was already unanchored and already
   covered `dev.db` at any depth, so the redundant explicit
   `backend/data/dev.db` line was simplified away.)

**Still open after this phase (nothing here could be closed without an
actual Docker daemon):**
- `docker compose up` itself has never been run. All four fixes above are
  confident-but-unverified: correct by inspection against how each piece
  of the real (tested) code actually behaves, but the very first real
  `docker compose up` in an environment with a daemon should still be
  treated as the true first test of this file, not a formality — watch
  specifically for anything Phase 11 review missed (e.g. exact Debian
  package names for a future Debian release, or an nginx default config
  quirk that only shows up when actually serving the built `dist/`).
  `QDRANT_MODE=server` against the real `qdrant` service in particular
  needs its own explicit check per the pre-existing KNOWN GAPS item 2
  below, since only `QDRANT_MODE=local` has ever been exercised in any
  session.
- The four disk/network-blocked gaps items 5/8/9 already describe
  (FlagEmbedding+torch, a real LLM API key) are completely orthogonal to
  Docker and remain exactly as blocked as before — Phase 11 didn't touch
  them.

### Phase 10 design & implementation summary

- **Backend test isolation (KNOWN GAPS item 1 / item 14a):** added
  `backend/tests/conftest.py`. It sets `DATABASE_URL`, `QDRANT_LOCAL_PATH`,
  `UPLOAD_DIR` and `PROCESSED_DIR` to a fresh `tempfile.mkdtemp()` tree
  *before* any `app.*` module is imported (conftest.py is always imported
  by pytest before the test modules in its directory, and the module-level
  code here is pure stdlib — no `app` import — specifically so this
  ordering is reliable), then removes that whole tree in a
  session-scoped `autouse` fixture's teardown. This only needed to cover
  Phases 1/2/4's test files: Phase 5 onward (`test_phase6_generation.py`,
  `test_phase7_reliability.py`, `test_phase8_conversations.py`) already had
  their own per-fixture `tmp_path` + `monkeypatch` overrides for this exact
  reason, so they're unaffected by (and don't conflict with) the new
  conftest. A per-session temp path was chosen over
  `sqlite:///:memory:` deliberately: `TestClient`'s FastAPI app is exercised
  across multiple threads/connections, and SQLite's `:memory:` DB is
  private to the single connection that created it, so a shared in-memory
  DB across connections would need `StaticPool`-style engine wiring that
  the existing `app/database/session.py` doesn't have — a real temp *file*
  sidesteps that with a one-line, low-risk change instead of touching
  production DB-engine code for a test-only concern. Verified: deleted
  `backend/data/dev.db`/`backend/data/qdrant` were NOT recreated by a full
  `pytest` run (confirmed by listing `backend/data/` immediately after),
  proving the suite no longer touches the real dev DB/vector store at all.
- **Frontend test runner (KNOWN GAPS item 14b):** added `vitest`,
  `@testing-library/react`, `@testing-library/jest-dom` and `jsdom` as
  devDependencies, plus `vitest.config.ts` (kept deliberately separate from
  `vite.config.ts` so `npm run build`'s `tsc -b && vite build` pipeline
  never has to know about test-only config, and vice versa) and
  `src/test/setup.ts` (just the jest-dom matcher import). Two test files,
  matching exactly the two places PROJECT_STATE.md's own item 14b called
  out as the ones a backend contract change could silently break:
  - `src/services/api.test.ts` — covers `handle<T>()`'s four branches by
    stubbing `global.fetch`: a 2xx body decodes as JSON; a non-2xx body's
    `detail` field becomes the thrown `Error`'s message (the exact 503
    shape `ChatMessage`'s error bubble depends on, per Phase 9's notes); a
    non-JSON error body falls back to `statusText`; a 204 response resolves
    to `undefined` without calling `.json()` at all (matters because a
    204's body is empty and calling `.json()` on it would throw).
  - `src/hooks/useChat.test.ts` — covers the optimistic-message state
    machine: an exchange with `response: null` appears synchronously on
    `send()` and `sending` flips true; on success the same exchange is
    patched in place with the resolved `ChatResponseOut` and
    `conversation_id` is captured into hook state; on rejection the
    exchange is patched with `error` instead and `sending` returns to
    false; a second `send()` call reuses the `conversation_id` captured
    from the first response (this is the exact behavior Phase 9's design
    notes describe for `useChat`); blank/whitespace-only input never calls
    `api.chat` at all; `reset()` clears both `exchanges` and
    `conversationId`. `api.ts` is mocked with `vi.mock` so these tests
    don't depend on `fetch`/a real backend at all.
- **Not done, out of scope for this phase:** no tests were added for React
  components (`ChatMessage`, `EvidencePanel`, `DocumentList`, etc.) — doing
  that well needs `@testing-library/react`'s render/interaction APIs
  exercised against real DOM output, which is a reasonable next increment
  but wasn't in the two specific gaps item 14 called out, and keeping this
  phase scoped to the two highest-value, contract-shaped state machines
  (network-error handling + optimistic UI state) matched the "at minimum"
  framing item 14 used. No backend test file was rewritten to remove its
  own `tmp_path`/`monkeypatch` overrides in favor of relying solely on the
  new conftest.py — they already work correctly and duplicating isolation
  logic is harmless, so leaving them untouched avoided risking a working
  test file for a purely cosmetic dedup.

### Phase 9 design & implementation summary

Design questions Phase 8's handoff (former item 13, now resolved) posed
for Phase 9:

- **Chat UI layout**: the existing Phase 1 dashboard's placeholder "Chat
  is coming in a later phase" panel (`pages/Dashboard.tsx`'s `<main>`) is
  replaced by a real two-pane chat: a scrollable message list
  (`components/chat/ChatMessage.tsx`, one bubble pair per turn) and a
  pinned bottom input (`components/chat/ChatInput.tsx`, Enter to send,
  Shift+Enter for a newline). Kept inside the existing sidebar+main
  layout and Tailwind palette (slate/indigo/emerald/amber/red) already
  established in Phase 1 rather than introducing a new visual language —
  this is an extension of one existing screen, not a new product, so
  consistency with what's already there mattered more than a fresh look.
- **Evidence viewer**: `components/chat/EvidencePanel.tsx`, toggled open
  per-answer via a "View sources (n)" link. Shows every passage
  `ChatResponseOut.sources` returned (not just cited ones — see
  `SourcePassageOut`'s own backend docstring on why), with cited passages
  visually distinguished (indigo left-tint + a "cited" pill) and any
  passages Phase 7 flagged as conflicting highlighted in red with the
  specific contradicting sentences shown inline.
  `components/chat/ConfidenceBadge.tsx` renders Phase 7's confidence
  level as a colored pill with the `reasons` list in a hover tooltip;
  `insufficient_evidence` answers get a distinct amber treatment instead
  of the normal white bubble.
- **`conversation_id` generation**: left entirely to the server. The
  frontend sends no `conversation_id` on a chat panel's first message;
  `useChat` (in `hooks/useChat.ts`) captures whatever id
  `ChatResponseOut.conversation_id` comes back with and sends that same
  id on every subsequent turn in that panel. This avoids needing any
  client-side id generation or persistence code at all, at the cost of a
  chat panel's history being lost on a full page reload (no
  `localStorage`/`sessionStorage` persistence was added — judged out of
  scope for this phase; `GET /api/conversations/{id}` already exists
  server-side if a future phase wants to rehydrate a saved id from
  storage). A document's selection change resets the chat panel's local
  `conversationId`/message list entirely (`ChatPanel`'s `useEffect` on
  `documentId` calls `useChat`'s `reset()`) — switching what you're
  asking about starts a fresh conversation rather than mixing scopes.
- **`resolved_query` UI treatment**: a small italic "Understood as:
  '...'" line shown above the answer, only rendered when
  `ChatResponseOut.resolved_query` is non-null (i.e. only on turns Phase
  8 actually rewrote) — exactly the transparency note the Phase 8 handoff
  asked about, no more.

Other Phase 9 additions, beyond the four questions above:
- `components/DocumentList.tsx` gained click-to-select (only for
  `processing_status === "completed"` documents — selecting an
  in-progress or failed document to scope chat against would just filter
  chat's retrieval down to a document with no indexed chunks yet), with a
  selected-row indigo highlight. Clicking the same document again
  deselects it back to "all documents."
  `pages/Dashboard.tsx` now holds `selectedDocId` state, clears it if the
  selected document is deleted or otherwise disappears from the list
  (effect watching `documents`), and passes the resolved document
  id/name down to `ChatPanel`.
- `types/chat.ts` (new): hand-mirrors
  `ChatRequest`/`SourcePassageOut`/`ConfidenceOut`/`ConflictPairOut`/
  `ConflictsOut`/`ChatResponseOut`/`ConversationTurnOut`/`ConversationOut`
  from `backend/app/schemas/schemas.py`, plus a frontend-only
  `ChatExchange` shape (one optimistic-then-resolved turn in a panel's
  message list) — same manual-sync convention `types/document.ts`
  already established in Phase 1.
- `services/api.ts` gained `chat()` and `getConversation()`, same
  `fetch` + `handle<T>()` error-unwrapping pattern as every existing
  method (a non-2xx response's JSON `detail` field, e.g. the 503 message
  above, becomes the thrown `Error`'s message — this is what
  `ChatMessage` displays verbatim in its error bubble).

**Known Phase 9 gaps, deliberately deferred:**
- No automated frontend tests (no test runner is configured in the
  scaffold at all yet — see new "KNOWN GAPS" item 14).
- Never actually seen a real assistant answer render end-to-end in a
  browser this session, since (a) this sandbox has no browser to drive
  and (b) `/api/chat` itself is still blocked on the same missing
  `FlagEmbedding`/`torch` + real `LLM_API_KEY` gaps Phases 4/6 already
  documented — confirmed unchanged this session (still 503s with the
  same message). What *was* verified live: the dev server actually
  proxies to a live backend, the 503/404 error paths render into the
  error UI exactly as coded (traced through `services/api.ts` ->
  `useChat` -> `ChatMessage` by inspection, not a rendered screenshot),
  and the full request/response TypeScript shapes match the real
  `ChatResponseOut`/`ConversationOut` JSON the live backend actually
  returns for the paths that don't need those two gaps closed (the 404
  and the 503's `detail` string). The first session with both a browser
  (or at least a way to screenshot a headless one) and a working
  `/api/chat` should do a real click-through of a real answer, including
  a real multi-turn follow-up and a real conflict-detection case, and
  fix anything that looks wrong that TypeScript/build success can't
  catch (spacing, wrapping, empty/loading states under real network
  latency instead of instant mock responses).

### Phase 8 design & implementation summary

Design questions PROJECT_STATE.md's own Phase 7 handoff (former item 12,
now resolved) posed for Phase 8, and how they were resolved — full
reasoning also lives in `app/conversations/memory.py`'s module docstring:

- **Storage**: a real DB table, not in-memory. Two new SQLAlchemy models
  in `app/database/models.py`: `Conversation` (id, created_at, optional
  document_id) and `ConversationTurn` (conversation_id, turn_index,
  raw_query, resolved_query, answer, created_at). Created lazily — a
  `/api/chat` call that never sends `conversation_id` creates nothing and
  costs nothing extra, preserving Phase 6/7's exact original stateless
  contract for callers who don't opt in.
- **History window**: the last `CONVERSATION_HISTORY_TURNS` turns (new
  `Settings` field, default 3) feed into follow-up resolution — not the
  whole conversation, so an old unrelated turn from far earlier can't leak
  into a much later rewrite.
- **Resolution mechanism**: one extra small LLM call
  (`resolve_followup()` in the new `app/conversations/memory.py`), not a
  rule-based rewrite — a regex/pronoun approach can't handle a fully
  elided follow-up like "what about last quarter?". This is the standard
  "condense question" step from production RAG systems. A new
  `CONVERSATION_REWRITE_ENABLED` setting (default True) can turn it off
  (still get conversation tracking/history, just no extra call). Skipped
  unconditionally on a conversation's first turn (nothing to resolve
  against yet, regardless of the setting), and degrades silently to the
  raw query (never raises, never 503s the request) if the rewrite call
  itself hits `LLMUnavailableError` or returns an empty string.

Files touched:
- `app/database/models.py`: added `Conversation`, `ConversationTurn`.
- `app/config.py`: added `CONVERSATION_HISTORY_TURNS`,
  `CONVERSATION_REWRITE_ENABLED`.
- `app/conversations/memory.py` (new): `ResolvedQuery` dataclass,
  `resolve_followup()`, `build_history_block()`,
  `get_or_create_conversation()`, `get_recent_turns()`, `record_turn()`.
- `app/schemas/schemas.py`: `ChatRequest.conversation_id` (new, optional);
  `ChatResponseOut.conversation_id`/`resolved_query` (new, both
  optional — `None` unless a conversation was tracked / an actual rewrite
  happened); new `ConversationTurnOut`/`ConversationOut`.
- `app/api/chat.py`: loads recent history and calls `resolve_followup()`
  when `conversation_id` is present, runs `generate_answer()` against the
  *resolved* query (not the raw one), records the turn afterward, and
  returns the new response fields. `ChatResponseOut.query` now always
  echoes the caller's raw `body.query` (previously it echoed whatever was
  passed into `generate_answer()`, which is now the resolved query, not
  the raw one, when a conversation is active — this could be a subtle
  behavior change for any existing caller inspecting `query` specifically
  while also using `conversation_id`, though no such caller exists yet).
- `app/api/conversations.py` (new): `GET /api/conversations/{id}`,
  404 if unknown. Read-only by design — a conversation's content is
  governed entirely by the `/api/chat` turns that created it, no direct
  POST/PATCH.
- `app/main.py`: registered the new `conversations` router (the roadmap
  comment for this was already there from project kickoff).
- `tests/test_phase8_conversations.py` (new, 17 tests): direct unit tests
  for `resolve_followup()`/`build_history_block()` with a fake LLM client
  (rewrite happens / skipped on first turn / skipped when disabled /
  passes through an already-standalone query / strips wrapping quotes /
  falls back on LLM failure / falls back on empty rewrite); direct DB
  tests for `get_or_create_conversation()`/`get_recent_turns()`/
  `record_turn()` against a real temp-file SQLite session; and
  `/api/chat` + `/api/conversations/{id}` integration tests against a
  real embedded/local Qdrant collection (same pattern as
  test_phase5/6/7), including one that runs two turns end-to-end and
  confirms the second turn's follow-up is actually rewritten before
  retrieval, the answer is generated from the rewritten query, and the
  full two-turn history round-trips correctly through the new GET
  endpoint.

This session's environment had a real `pip`, normal disk (~10 GB free),
and **real network access to PyPI** (unlike the Phase 5/6 sessions —
`huggingface.co` is still not reachable, so the FlagEmbedding/torch/
real-model gaps below are unchanged). Per the handoff instructions at the
top of this file, the very first thing done this session, before any
Phase 7 code:
1. `pip install -r backend/requirements.txt` — succeeded completely (all
   packages installed, including `paddleocr`/`paddlepaddle`/`qdrant-client`
   — `FlagEmbedding`/`torch` remain commented out per the file's own
   Phase 4 notes, disk still not confirmed sufficient for `torch`, not
   re-tested this session since it wasn't blocking Phase 7).
2. `pytest backend/tests/ -q` — **all 84 pre-existing tests passed**,
   confirming Phases 1-6 together for the first time ever (Phase 5's
   `test_phase5_retrieval.py` and Phase 6's `test_phase6_generation.py`
   had never been run against real dependencies before this session).
   This resolves the biggest open item from the Phase 6 handoff.

Phase 7's two logic modules (`app/reliability/confidence.py`,
`app/reliability/conflict.py`) already existed in the uploaded project
(written in an earlier, disconnected session — same pure-Python,
zero-dependency style as `citation.py`/`prompt.py`) but were **not yet
wired into `generator.py`/`chat.py`/`schemas.py`, and had no tests**. This
session:
- Added `confidence: ConfidenceResult` and `conflicts: ConflictDetectionResult`
  fields to `generator.py`'s `GeneratedAnswer`, computed on both the
  zero-evidence short-circuit path and the normal LLM-call path.
- Added `ConfidenceOut`/`ConflictsOut`/`ConflictPairOut` to `schemas.py`
  and wired them into `ChatResponseOut` + `app/api/chat.py`'s response
  construction.
- Wrote `tests/test_phase7_reliability.py` (18 tests): direct unit tests
  for `compute_confidence()`/`detect_conflicts()` (pure Python, no
  mocking), plus three `/api/chat` integration tests against a real
  embedded/local Qdrant collection (same pattern as test_phase5/6),
  including one that seeds two chunks with a genuine numeric contradiction
  and confirms the endpoint actually reports it.

---

## VERIFICATION LOG (Phase 7)

Real bugs found and fixed this session by actually running the new tests
against real code — not just written and assumed correct:

1. **`conflict.py`'s number regex matched digits glued to a preceding
   letter** (e.g. the "3" in "Q3"), extracting a bogus standalone numeric
   fact that could then be compared against an unrelated number elsewhere
   and reported as a false-positive conflict. First caught by
   `test_detect_conflicts_no_conflict_when_values_agree_within_tolerance`
   (expected no conflict, got one). Fixed with a negative lookbehind
   (`(?<![A-Za-z0-9])`) so a digit immediately preceded by a letter or
   digit is no longer treated as the start of a standalone number. All
   other conflict-detection behavior (magnitude suffixes, percent vs.
   dollar exclusion, same-passage exclusion, Jaccard context-overlap
   threshold) was exercised directly and needed no changes.
2. **Test-authoring bugs, not code bugs** (both in this session's own new
   test file, caught and fixed the same way): an assertion that checked
   for the literal substring `"insufficient"` in a reason string that
   actually reads "...don't contain enough information..." (wording
   mismatch, trivial fix), and an assumption that a weak (0.05) reranked
   retrieval score combined with a perfectly-cited answer should be
   "low" confidence — `compute_confidence()`'s documented 50/50 weighting
   actually produces "medium" (~0.53) in that case, which is correct
   per the module's own design, not a bug; the test's expected value was
   corrected instead of the code.
3. **Test fixture bug**: an integration test called `get_settings()`
   directly to fetch the Qdrant path/collection it had just configured via
   `app.dependency_overrides` — but `dependency_overrides` only intercepts
   calls FastAPI's own DI resolves for an endpoint parameter, not a direct
   Python call to the same function elsewhere in test code, so this
   returned the *default* settings, not the test's overridden ones. Fixed
   by having the `client` fixture stash the actual `Settings` instance it
   configured (`client.test_settings`) for tests to read back directly.
4. **`get_qdrant_client()`'s process-wide singleton wasn't reset between
   tests in the new file** (other test files in this project already do
   this — this file initially didn't). Local-mode `QdrantClient` locks its
   on-disk directory to one open client; without resetting the cached
   singleton, a later test's request silently reused an earlier test's
   client (pointed at that earlier test's now-torn-down `tmp_path`),
   producing "zero results" even though the current test's data was
   correctly seeded on disk. Fixed by calling
   `app.retrieval.qdrant_store.reset_qdrant_client_cache()` before and
   after each test in the `client` fixture, matching the reset pattern
   this project already uses elsewhere for the same reason.

After these four fixes, all 18 new Phase 7 tests pass, and the full suite
(102 tests total: 84 pre-existing + 18 new) passes together. This is
real, run verification — not a description of intended behavior.

**Still NOT verified this session** (unchanged gaps, need real
`huggingface.co` network access and/or enough disk for `torch`, or a real
LLM API key — none of these blocked Phase 7 since Phase 7's own logic has
zero model/HTTP dependency):
- The real PaddleOCR-VL model (Phase 3) and real bge-m3 embeddings /
  bge-reranker-v2-m3 (Phase 4/5) have still never run against real
  weights in any session.
- `llm_client.py`'s real HTTP call shapes are still only verified against
  a monkeypatched `httpx.post`, never a real provider — no `LLM_API_KEY`
  supplied to any session yet.
- `docker-compose.yml` still not run end-to-end (no Docker daemon in this
  sandbox either).

See "KNOWN GAPS" below for the updated, prioritized list for whichever
session finally has `huggingface.co` access, enough disk for `torch`, a
real LLM API key, and/or a Docker daemon.

---


**Note on the list below (kept for history, superseded by this session):**
the Phase 5/6 sessions both had zero outbound network access at all,
which is why they read as pessimistic about ever installing anything.
This session had real PyPI access (see "VERIFICATION LOG (Phase 7)"
above) but still not `huggingface.co`, so gap #1 (OCR/embedding model
downloads) and half of gap #3 (real LLM HTTP shapes — no API key
supplied) are still open; gap #2 (no network at all) is now resolved.
Gap #4 is unaffected either way.

1. Phase 3's OCR model and Phase 4's embedding model can't reach
   `huggingface.co` (network allow-list in some sessions) — Phase 4
   additionally found `torch` itself can't fit on disk (a few GB free vs.
   >10 GB needed) in that session; this session had ~10 GB free but did
   not attempt installing `torch`/`FlagEmbedding` since it wasn't blocking
   Phase 7 — still worth trying in a session that has both
   `huggingface.co` access and confirmed disk headroom.
2. ~~This session (Phase 6), exactly like Phase 5's session, had no
   outbound network access whatsoever~~ — **resolved as of Phase 7**: this
   session had real PyPI access and `pip install -r backend/requirements.txt`
   succeeded completely (see "VERIFICATION LOG (Phase 7)" above).
3. `app/generation/llm_client.py`'s real HTTP call shapes are still
   **not** confirmed against a real provider response — this session had
   PyPI access but no `LLM_API_KEY` for any provider, so verification
   stayed at the `httpx.post`-monkeypatch level established in Phase 6
   (see test_phase6_generation.py). Still the top item for whichever
   session finally has a real key.
4. Earlier phases flagged that only `api.anthropic.com` was ever confirmed
   reachable in some sandbox sessions (network allow-list, not "no network
   at all"). This session's allow-list included `pypi.org` and friends but
   NOT `huggingface.co` or any LLM provider host — consistent with "an
   allow-list, not global access." The earlier note about a possible
   Anthropic-API `LLM_PROVIDER` option remains worth considering if
   DeepSeek/OpenAI/Gemini's hosts turn out unreachable in a future session
   while `api.anthropic.com` is.

Phase 6's code was written under a no-network constraint and evaluated
mostly by pure-Python execution + careful reading at the time; this
session's real pytest run (see "VERIFICATION LOG (Phase 7)" above) is the
first genuine confirmation that Phase 6's generator/endpoint code
actually works, not just reads correctly.

---

## VERIFIED DECISIONS (do not re-derive these — they were checked against
## live documentation/registries during this build; only re-verify if a
## phase's install step fails)

- **PaddleOCR-VL-1.6** official install (verified against the HuggingFace
  model card, June 2026):
  ```bash
  # GPU (CUDA 12.6 example — adjust per https://www.paddlepaddle.org.cn/en/install/quick):
  python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
  # CPU-only:
  python -m pip install paddlepaddle==3.2.1
  python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
  ```
  Python API:
  ```python
  from paddleocr import PaddleOCRVL
  pipeline = PaddleOCRVL(pipeline_version="v1.6")
  output = pipeline.predict("path/to/page.png")
  for res in output:
      res.save_to_json(save_path="output")
      res.save_to_markdown(save_path="output")
  ```
  macOS has no native Paddle wheel path documented — use Docker on macOS.
  Architecture is compatible with PaddleOCR-VL-1.5 (drop-in upgrade path).

- **Backend stack pinned (Phase 1, installed & tested):** fastapi 0.115.6,
  uvicorn 0.34.0, pydantic 2.10.4, pydantic-settings 2.7.1, sqlalchemy 2.0.36,
  python-dotenv 1.0.1, loguru 0.7.3, pytest 8.3.4, httpx 0.28.1. Python 3.12.

- **Frontend stack pinned (Phase 1, installed & tested):** Vite (react-ts
  template), React 19, TypeScript, Tailwind CSS v4 via `@tailwindcss/vite`
  plugin (NOT the old `tailwind.config.js` + PostCSS setup — v4 uses
  `@import "tailwindcss";` directly in `index.css` and a Vite plugin).

- **DB:** SQLite for dev (`sqlite:///./data/dev.db`), Postgres for
  docker-compose/production, switched purely via `DATABASE_URL` env var —
  no code changes needed (see `backend/app/database/session.py`).

- **Normalized schema is locked in** (`backend/app/database/models.py`):
  `Document` → `DocumentPage` → `DocumentElement` → `Chunk`. All later
  phases write into these same four tables. Do not redesign this schema
  without updating this file.

- **Phase 2 ingestion/chunking design (locked in, verified working —
  see "Verification Log (Phase 2)" below):**
  - Scanned-PDF detection: a PDF is treated as `scanned_pdf` (routed to a
    Phase 3 OCR holding status, not ingested as near-empty text) when its
    average extractable text per page is below `SCANNED_PDF_MIN_CHARS_PER_PAGE`
    (default 20). This lives in `app/ingestion/pdf_extractor.py`.
  - Chunking: naive fixed-size sliding window, **per page** (chunks never
    cross a page boundary, so every `Chunk` row cites exactly one
    `page_number` — important for Phase 6 source attribution). Defaults:
    `CHUNK_SIZE_CHARS=1000`, `CHUNK_OVERLAP_CHARS=150`. Lives in
    `app/chunking/naive_chunker.py`. Swap for something smarter later
    without changing the `Chunk` table shape.
  - Processing status flow for a normal text PDF: `uploaded` → `detecting`
    → `processing` → `extracting` (pages written) → `chunking` (chunks
    written) → `completed`. A scanned PDF or image stops at `ocr` (holding
    status, not yet actually OCR'd — that's Phase 3). `.docx`/`.xlsx`/`.csv`
    currently go straight to `failed` with a "not implemented yet" message
    (no phase claims them yet in the table below; add one if the person
    wants office-doc support).
  - Ingestion runs as a `BackgroundTasks` call from
    `POST /api/documents`, added in `app/ingestion/router.py`, using its
    own DB session (background tasks in FastAPI run in a thread pool, so
    they can't reuse the request's session).
  - Two debug endpoints added: `GET /api/documents/{id}/pages` and
    `GET /api/documents/{id}/chunks` — not asked for explicitly, but cheap,
    useful for verifying Phase 2 by hand, and likely reusable by the
    Phase 9 evidence viewer.

- **Phase 3 OCR design (locked in, verified — see "Verification Log
  (Phase 3)" below):**
  - Scanned PDFs are rasterized page-by-page to PNGs (`app/ocr/rasterizer.py`,
    PyMuPDF `get_pixmap`, default `OCR_RENDER_DPI=200`) into
    `data/processed/{document_id}/pages/`, then each page image (and any
    standalone image upload, same code path) goes through OpenCV
    preprocessing (`app/ocr/preprocessing.py`: grayscale, denoise, skew
    correction — skipped/no-op if skew is negligible or unmeasurable) before
    OCR. Toggle via `OCR_ENABLE_PREPROCESSING`.
  - `app/ocr/paddle_ocr.py` is the only place that touches the `paddleocr`
    package. `get_ocr_pipeline()` is a single lazy-singleton factory;
    `run_ocr_on_images()` takes the pipeline as a parameter rather than
    creating it itself. This is deliberate: it's the seam tests monkeypatch
    (`app.ingestion.router.get_ocr_pipeline`) to inject a fake pipeline
    that mimics the real output shape, since the real model can't download
    in this sandbox (see the network-access warning above). **Follow this
    same pattern for Phase 4/5/6's model/API calls.**
  - Output parsing depends on the real `paddlex` package's documented shape
    (confirmed by reading the installed package's own source, since the
    pipeline itself couldn't be run): each page result is dict-like with a
    `parsing_res_list` key — a list of blocks with `.label` ("text",
    "paragraph_title", "table", "formula", "chart", "image", "seal", ...),
    `.content` (text, or an HTML table string for `"table"`), `.bbox`
    (`[x0,y0,x1,y1]`). `_classify_block()` maps these labels to our own
    `element_type` vocabulary (`paragraph | heading | table | formula |
    image | list`). A malformed/unparseable table HTML never raises — it
    falls back to keeping the raw content as text with `table_data=None`.
  - `OCRUnavailableError` is the one exception type both `get_ocr_pipeline()`
    and `run_ocr_on_images()` raise on any external-boundary failure (missing
    package, model init failure, a `predict()` crash mid-run). The router
    catches exactly this type and marks the document `failed` with a clear
    message — nothing else in `_process_via_ocr` is allowed to silently
    swallow an OCR failure.
  - `DocumentElement` rows are now actually written (Phase 2 only wrote
    `DocumentPage`/`Chunk`): one row per OCR block, with `bbox`, `confidence`,
    and `table_data` populated where available. `element_metadata` currently
    just stores `{"order": <reading-order index>}`.
  - Chunking OCR output (`_chunk_ocr_pages` in `app/ingestion/router.py`)
    is NOT the same function as Phase 2's `chunk_pages` — it reuses the same
    underlying `split_text_sliding_window()` (factored out of
    `app/chunking/naive_chunker.py` for this reason) for prose, but treats
    every `table` element as its own single whole chunk (`element_type=
    "table"`), never split by the sliding window and never merged with
    surrounding prose — this matters for the demo questions that need a
    clean, complete table in one chunk (e.g. "highest revenue year in the
    table").
  - Status flow for scanned PDF / image is now: `uploaded` → `detecting` →
    (scanned PDF only: `processing` to run PyMuPDF's scan-detection first)
    → `ocr` → `extracting` (pages+elements written) → `chunking` → `completed`,
    or `failed` at the `ocr` step if the pipeline is unavailable. This
    replaces Phase 2's old permanent `ocr` holding status.

- Embedding/reranker/LLM choices are **configurable via env vars only**
  (see `.env.example`) — nothing is hard-coded, per project requirements.
  Defaults chosen so far (not yet wired into code): `EMBEDDING_MODEL=BAAI/bge-m3`,
  `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`, `LLM_PROVIDER=deepseek`.
  These are defaults only — swap freely in Phase 4/5/6 if a better option
  turns up; nothing downstream depends on these specific choices yet.

- **Phase 4 embeddings/indexing design (locked in, verified — see
  "Verification Log (Phase 4)" below):**
  - `app/embeddings/embedder.py` is the only place that touches
    `FlagEmbedding`. Same lazy-singleton/boundary pattern as Phase 3's
    `paddle_ocr.py`: `get_embedding_model()` creates (and caches) the
    model, `embed_texts()` takes the model as a parameter, and both raise
    `EmbeddingUnavailableError` on any external-boundary failure (package
    not installed, model init failure, an `encode()` crash mid-batch).
    Tests inject a fake model at this boundary (see
    `tests/test_phase4_embeddings.py`), since the real model can't even be
    installed here (see the "⚠️ IMPORTANT" warning above).
  - Real API shape (`BGEM3FlagModel(model_name_or_path, use_fp16=bool,
    devices=str|list|None, ...)`, `.encode()` returning `{"dense_vecs":
    np.ndarray, "lexical_weights": [...], "colbert_vecs": [...]}`) was
    confirmed by downloading the `FlagEmbedding` wheel with `pip download
    --no-deps` (no torch/install needed for that) and reading
    `FlagEmbedding/inference/embedder/encoder_only/m3.py` directly — this
    caught a real bug before it shipped: the constructor parameter is
    `devices` (plural), not `device`.
  - Only dense vectors (1024-dim, `BGE_M3_DIMENSION` in `embedder.py`) are
    produced/indexed in Phase 4. `EmbeddingResult.sparse` and
    `Chunk`-payload plumbing already anticipate Phase 5 requesting
    `return_sparse=True` from the same bge-m3 call to add lexical/sparse
    vectors for hybrid retrieval — no schema changes needed then, just wire
    the extra field through.
  - `app/retrieval/qdrant_store.py` is the only place that touches
    `qdrant-client`. `QDRANT_MODE=local` (default) opens an embedded,
    on-disk Qdrant collection needing no server/network at all — this is
    what dev/tests actually use and is genuinely, fully verified end to
    end (no mocking needed for this half of Phase 4, unlike the embedding
    model). `QDRANT_MODE=server` talks to a real Qdrant instance
    (docker-compose's `qdrant` service) via the identical client API, but
    that mode itself hasn't been exercised against a live server in this
    sandbox — Phase 11 (Docker Compose) should do one real round-trip
    against it before trusting it in production.
  - Qdrant point IDs must be a valid UUID or unsigned int (a real,
    reproduced constraint of local-mode Qdrant, hit while writing this
    phase's own tests) — `Chunk.id` is already a UUID string
    (`app/database/models.py`), so it's reused directly as the point ID
    rather than tracking a second ID.
  - Ingestion status flow now ends `... -> chunking -> embedding ->
    indexing -> completed` for BOTH the PyMuPDF and OCR pipelines (was
    `... -> chunking -> completed` before Phase 4) — one shared function,
    `_embed_and_index_chunks()` in `app/ingestion/router.py`, is called
    from both `_process_pdf` and `_process_via_ocr` once their respective
    chunks are written, so retrieval/generation in later phases never
    needs to know which pipeline produced a given chunk. On an
    `EmbeddingUnavailableError`, the document is marked `failed` with a
    clear message at the `embedding` step, exactly like Phase 3's
    `OCRUnavailableError` handling one step earlier in the same pipeline.
  - `DELETE /api/documents/{id}` now also calls
    `delete_document_vectors()` so removing a document doesn't leave
    orphaned vectors behind in Qdrant (best-effort — a Qdrant cleanup
    failure logs a warning but does not block the delete).

- **Phase 5 hybrid retrieval design (written this session, NOT
  execution-verified — see "⚠️ IMPORTANT" above and "Verification Log
  (Phase 5)" below):**
  - `app/embeddings/embedder.py`: `embed_texts()` now requests
    `return_sparse=True` by default and parses bge-m3's `lexical_weights`
    output into `EmbeddingResult.sparse` (`list[dict[str, float]]`,
    token-id-string -> weight) — the field Phase 4 left unused for exactly
    this reason. `embed_query(text, model)` is a new convenience wrapper
    for the single-query case, returning `(dense_vector, sparse_weights)`.
  - `app/retrieval/qdrant_store.py`: the collection schema changed from
    Phase 4's single unnamed dense vector to **named vectors** —
    `DENSE_VECTOR_NAME="dense"` (1024-dim, cosine, same as before) plus
    `SPARSE_VECTOR_NAME="sparse"` (bge-m3 lexical weights, stored as a
    Qdrant `SparseVector`). This is a breaking schema change from Phase 4;
    nothing worth migrating exists in this sandbox's dev-only local Qdrant,
    but a real Phase-4 deployment would need to recreate/reindex its
    collection when upgrading. `upsert_chunks()` gained an optional
    `sparse_vectors` parameter (omit/`None` still writes valid dense-only
    points — sparse search simply won't match them, which is the correct
    degrade path). `search()` is kept as a backward-compatible alias for
    the new `search_dense()`; `search_sparse()` is the new sparse leg.
    `_to_sparse_vector()` converts bge-m3's `{token_id_str: weight}` dict
    into Qdrant's `SparseVector(indices=[...], values=[...])`, skipping
    (not raising on) any non-integer token id defensively.
  - `app/retrieval/fusion.py`: plain-Python Reciprocal Rank Fusion
    (`reciprocal_rank_fusion(dense_hits, sparse_hits, k=RRF_K)`) — no
    external dependency, deliberately not using Qdrant's native
    server-side fusion query so this piece has no unverified client-API
    surface to trust. **This is the one Phase 5 module that was actually
    exercised for real this session** (see Verification Log) since it has
    zero dependencies.
  - `app/retrieval/reranker.py`: `BAAI/bge-reranker-v2-m3` wrapper,
    same lazy-singleton/boundary pattern as `embedder.py` /
    `paddle_ocr.py` (`get_reranker_model()` factory,
    `RerankerUnavailableError` on any external-boundary failure, `rerank()`
    takes the model as a parameter). **Read this file's own docstring
    before touching it** — its real API shape (`FlagAutoReranker.from_finetuned(...)`
    / `.compute_score(pairs, normalize=...)`) is from documented
    `FlagEmbedding` knowledge, NOT confirmed by reading the installed
    package's source the way Phase 4 confirmed `BGEM3FlagModel`'s shape,
    because this session had no network access at all — not even for the
    `pip download --no-deps` trick that worked in Phase 4. Treat the exact
    method name/kwargs as unverified until someone can run the Phase
    4-style source-reading pass.
  - `app/retrieval/hybrid.py`: `hybrid_search()` is the one function later
    phases should actually call — wires `embed_query()` -> `search_dense()`
    + `search_sparse()` -> `reciprocal_rank_fusion()` -> optional
    `rerank()`. Degrades gracefully: an unavailable embedding model
    re-raises (search is impossible without a query vector — caller should
    500/503), but an unavailable/crashing reranker is caught internally and
    falls back to the fused RRF order, logging a warning — reranking is a
    quality improvement, not a correctness requirement.
  - `app/api/search.py`: new `POST /api/search` endpoint (`{"query": str,
    "document_id": str | None}` -> ranked results with score, reranked
    flag, dense/sparse ranks, chunk text, and source metadata) — the same
    "debug/inspection endpoint before the next phase's real consumer"
    role Phase 2's `/pages`/`/chunks` played for ingestion. Looks up the
    embedding model (hard-fails 503 on `EmbeddingUnavailableError`) and
    reranker model (soft-fails, proceeds without it on
    `RerankerUnavailableError`) itself, then delegates to `hybrid_search()`.
  - New config: `RERANKER_DEVICE`, `RERANK_ENABLED` (kill switch for the
    reranker without a redeploy), `RETRIEVAL_RERANK_POOL` (candidates sent
    to the reranker, smaller than `RETRIEVAL_TOP_K_CANDIDATES` since
    reranking is the expensive step), `RRF_K` (standard RRF constant,
    default 60, not usually worth tuning).
  - Phase 2/3/4's fake embedding models (`_FakeEmbeddingModel` in
    `test_phase2_ingestion.py`, `test_phase3_ocr.py`,
    `test_phase4_embeddings.py`) were all updated to also return
    `lexical_weights` alongside `dense_vecs`, since `embed_texts()` now
    requests sparse output by default and would otherwise get a
    length-mismatched (empty) sparse list. `app/ingestion/router.py`'s
    `_embed_and_index_chunks()` also got a defensive guard: if
    `embed_texts()` ever returns a sparse list whose length doesn't match
    the chunk batch (shouldn't happen against a real bge-m3 call, but
    cheap to guard), it passes `sparse_vectors=None` to `upsert_chunks()`
    rather than letting a length-mismatch `ValueError` fail the whole
    embedding step over just the sparse leg.

- **Phase 6 LLM generation design (locked in, PARTIALLY verified — see
  "Verification Log (Phase 6)" below for exactly which half):**
  - `app/generation/prompt.py`: builds a fixed `SYSTEM_PROMPT` (answer only
    from the numbered context passages given; if insufficient, reply with
    a literal `INSUFFICIENT_EVIDENCE: <reason>` prefix; every factual
    sentence must end with `[n]` citation marker(s); never cite a number
    not given) plus a per-request user prompt numbering
    `hybrid_search()`'s results 1-based as `ContextPassage`s. Pure string
    formatting, zero dependency — **genuinely exercised this session**,
    not just written (see Verification Log).
  - `app/generation/citation.py`: parses every `[n]` marker out of the
    LLM's answer and validates each against the real passage count passed
    in (`num_passages`) — never trusts the LLM to have followed its own
    instructions. Distinguishes two failure modes on purpose (Phase 7 will
    likely want to weight these differently in a confidence score): an
    **invalid citation** (a cited number that does't correspond to any
    real passage — hallucinated reference) vs. an **uncited sentence** (a
    factual-looking sentence with no `[n]` marker at all — used the
    context but didn't attribute it). An `INSUFFICIENT_EVIDENCE:`-prefixed
    answer short-circuits to `fully_cited=True` — there's nothing to cite,
    and that's the correct non-hallucinating behavior, not a citation
    failure. Pure regex/string logic, zero dependency — **genuinely
    exercised this session** against 6 real scenarios (fully cited,
    hallucinated citation, uncited sentence, insufficient-evidence
    short-circuit, case-insensitivity of the prefix check, zero passages)
    — see Verification Log for the exact commands and output.
  - `app/generation/llm_client.py`: raw REST via `httpx` (no per-provider
    SDK — see the module's own docstring for why), one small class per
    provider family (`_OpenAICompatibleClient` covers both `deepseek` and
    `openai`, since DeepSeek's API is explicitly OpenAI-compatible;
    `_GeminiClient` for Gemini's different `generateContent` shape), same
    lazy-singleton/boundary pattern as every earlier external-model
    wrapper (`get_llm_client()` factory, one `LLMUnavailableError`
    exception type). Cached per `(provider, model, api_key)` tuple rather
    than unconditionally, since `LLM_PROVIDER` is explicitly documented as
    user-switchable at runtime (unlike the OCR/embedding/reranker
    singletons, which only ever have one real config per process). **NOT
    run this session** — `httpx` itself isn't installed here, so this
    module could not even be imported, let alone have an HTTP call made
    or mocked. Its request/response shapes follow each provider's public
    REST docs but are unconfirmed by source-reading (unlike Phase 4's
    `BGEM3FlagModel`) — see its own docstring, and item 3 under
    "⚠️ IMPORTANT" above.
  - `app/generation/generator.py`: `generate_answer()` is the one function
    later phases should call — wires `hybrid_search()` (Phase 5) ->
    `to_context_passages()` + `build_user_prompt()` -> `llm_client.complete()`
    -> `validate_citations()`. **Zero-results short-circuit**: if
    `hybrid_search()` returns no chunks at all, this returns a fixed
    `INSUFFICIENT_EVIDENCE:` answer directly and never calls the LLM —
    deliberate, not an optimization: it removes any chance of the model
    answering from outside knowledge when given empty context, which
    would silently violate the whole project's "grounded answer"
    requirement. A `document_id`-scoped search that comes up empty gets a
    message naming the document specifically (a different, more
    actionable outcome than "nothing in the whole corpus"). Unlike the
    reranker, an `LLMUnavailableError` is NOT caught here — it propagates
    to the caller (`app/api/chat.py`), which turns it into a 503, because
    there is no fallback path for "no LLM" the way there is for "no
    reranker." **NOT run this session** — importing this module requires
    the full Phase 4/5 dependency chain (`app.config` alone needs
    `pydantic-settings`; `hybrid_search`'s import chain needs `numpy`,
    `qdrant-client`, `qdrant_client.http.models`), none of which are
    installed here. (A hand-written `loguru`/`qdrant_client` stub pair was
    tried, same trick Phase 5 used for `loguru` alone, but the dependency
    surface here is wider than Phase 5's reranker/embedder isolation —
    `app.config`'s `pydantic_settings` import alone blocks it — so this
    was abandoned as not worth the stubbing effort versus just being
    honest that it's unverified; see Verification Log.)
  - `app/api/chat.py`: new `POST /api/chat` endpoint (`{"query": str,
    "document_id": str | None}` -> answer text, `insufficient_evidence`,
    `fully_cited`, `invalid_citation_indices`, and the full list of source
    passages offered to the LLM with a `cited: bool` per passage — the
    frontend's Phase 9 evidence viewer needs every offered passage, not
    just the cited subset, so a user can check what was available).
    Mirrors `app/api/search.py`'s error-boundary pattern, extended one
    step: both `EmbeddingUnavailableError` and `LLMUnavailableError` become
    a clear 503; `RerankerUnavailableError` is still soft-failed exactly
    like Phase 5. **NOT run this session** (needs FastAPI + the whole
    stack, same as `search.py`'s own endpoint tests in Phase 5).
  - New config in `app/config.py`: `LLM_MAX_TOKENS` (1024),
    `LLM_TEMPERATURE` (0.0 — deterministic by default, this is grounded
    Q&A not creative writing), `LLM_TIMEOUT_SECONDS` (60),
    `LLM_DEEPSEEK_BASE_URL` / `LLM_OPENAI_BASE_URL` / `LLM_GEMINI_BASE_URL`
    (all three always defined so switching `LLM_PROVIDER` never requires
    adding a new env var). New schemas in `app/schemas/schemas.py`:
    `ChatRequest`, `SourcePassageOut`, `ChatResponseOut`. `httpx` promoted
    from a dev/test-only dependency to a real runtime one in
    `requirements.txt` (same 0.28.1 already pinned — no version change,
    just a comment update, since it's still not actually installed here
    either way).

---

## PHASE-BY-PHASE STATUS

| Phase | Description | Status |
|---|---|---|
| 1 | Project setup: backend + frontend skeleton, health check, document upload/list/delete API, document library UI | ✅ Done & verified |
| 2 | Normal PDF ingestion (PyMuPDF), metadata, naive chunking | ✅ Done & verified |
| 3 | PaddleOCR-VL-1.6: scanned PDF routing, image OCR, layout/table extraction | ✅ Done & verified (OCR model call itself faked in tests — real weights unreachable in this sandbox, see warning above; everything else real) |
| 4 | Embeddings + Qdrant indexing | ✅ Done & verified (embedding model call itself faked in tests — real weights/torch unreachable in this sandbox, disk AND network, see warning above; Qdrant indexing/search/delete fully real, no mocking) |
| 5 | Hybrid retrieval (dense+sparse+fusion) + reranking | ⚠️ Code written, NOT run/verified in any session yet (no network at all in either session since — see "⚠️ IMPORTANT" above and Verification Log below) |
| 6 | LLM grounded answer generation + source attribution + citation validation | ⚠️ Code written; prompt/citation logic (pure Python) genuinely run+verified this session, LLM HTTP call + full endpoint NOT run (no network/deps at all — see Verification Log (Phase 6)) |
| 7 | Reliability: confidence, insufficient-evidence, conflict detection | ⏳ Not started |
| 8 | Conversation memory / follow-up question resolution | ⏳ Not started |
| 9 | Full frontend: chat UI, evidence viewer, processing status stepper | ⏳ Not started |
| 10 | Tests + evaluation dataset/script | ⏳ Partial (Phase 1 unit tests only) |
| 11 | Docker Compose full stack | ⏳ Compose file + Dockerfiles written, NOT yet run end-to-end |
| 12 | Documentation (README polish, screenshots, architecture diagram) | ⏳ README skeleton only |

---

## WHAT EXISTS RIGHT NOW (file inventory)

```
document-intelligence/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, lifespan init_db, /api/health
│   │   ├── config.py                # Settings (env-driven), all future config already defined
│   │   ├── api/documents.py         # POST/GET/GET-one/DELETE /api/documents (validated, sanitized)
│   │   ├── database/session.py      # engine/session/get_db/init_db
│   │   ├── database/models.py       # Document, DocumentPage, DocumentElement, Chunk
│   │   ├── schemas/schemas.py       # DocumentOut, DocumentPageOut, ChunkOut, HealthOut, ErrorOut
│   │   ├── ingestion/pdf_extractor.py  # PyMuPDF extraction + scanned-PDF detection (Phase 2)
│   │   ├── ingestion/router.py      # dispatches by extension, runs as BackgroundTask (Phase 2, extended Phase 3)
│   │   ├── chunking/naive_chunker.py   # fixed-size sliding-window chunker (Phase 2; split_text_sliding_window() factored out in Phase 3 for OCR reuse)
│   │   ├── ocr/paddle_ocr.py        # PaddleOCR-VL wrapper, output parsing, table extraction (Phase 3)
│   │   ├── ocr/rasterizer.py        # PDF page -> PNG rendering via PyMuPDF (Phase 3)
│   │   ├── ocr/preprocessing.py     # OpenCV denoise + deskew (Phase 3)
│   │   ├── embeddings/embedder.py   # BAAI/bge-m3 wrapper via FlagEmbedding (Phase 4; Phase 5 added sparse/return_sparse + embed_query())
│   │   ├── retrieval/qdrant_store.py  # Qdrant client, named dense+sparse vectors, upsert/search_dense/search_sparse/delete (Phase 4, Phase 5 restructured to named vectors)
│   │   ├── retrieval/fusion.py      # Reciprocal Rank Fusion, pure Python (Phase 5) — logic-verified this session, see below
│   │   ├── retrieval/reranker.py    # BAAI/bge-reranker-v2-m3 wrapper via FlagEmbedding (Phase 5) — NOT source-confirmed this session, see its docstring
│   │   ├── retrieval/hybrid.py      # hybrid_search() orchestrator: embed -> dense+sparse search -> RRF -> rerank (Phase 5)
│   │   ├── api/search.py            # POST /api/search debug/inspection endpoint wrapping hybrid_search() (Phase 5)
│   │   ├── generation/prompt.py     # SYSTEM_PROMPT + numbered-context user-prompt building (Phase 6) — real-verified this session
│   │   ├── generation/citation.py   # [n] citation extraction + validation against real passage count (Phase 6) — real-verified this session
│   │   ├── generation/llm_client.py # raw-REST deepseek/openai/gemini client, LLMUnavailableError boundary (Phase 6) — NOT run, no httpx here, see its docstring
│   │   ├── generation/generator.py  # generate_answer(): hybrid_search -> prompt -> LLM -> citation validation, zero-results short-circuit (Phase 6) — NOT run, needs full Phase 4/5 dep chain
│   │   ├── api/chat.py              # POST /api/chat wrapping generate_answer() (Phase 6) — NOT run, needs FastAPI+stack
│   │   └── (reliability/
│   │        conversations/ — still empty packages,
│   │        ready for Phase 7+ to fill in)
│   ├── tests/test_phase1_api.py     # 5 tests, verified passing (Phase 1 session)
│   ├── tests/test_phase2_ingestion.py  # 8 tests, verified passing as of Phase 4 session; fake embedding model updated again in Phase 5 (added lexical_weights) but NOT re-run this session — see Verification Log (Phase 5)
│   ├── tests/test_phase3_ocr.py     # 11 tests, verified passing as of Phase 4 session; same Phase 5 fixture update, same NOT-re-run caveat
│   ├── tests/test_phase4_embeddings.py  # 14 tests, verified passing as of Phase 4 session; fake embedding model updated in Phase 5 (added lexical_weights) but NOT re-run this session
│   ├── tests/test_phase5_retrieval.py  # 22 new tests written in the Phase 5 session — NOT run (no installable dependencies that session, see Verification Log (Phase 5))
│   ├── tests/test_phase6_generation.py  # new this session — citation/prompt tests genuinely pass (run directly, see Verification Log (Phase 6)); generator/endpoint tests written but NOT run (need the full Phase 4/5 dependency chain, unavailable this session too)
│   ├── requirements.txt             # Phase 1-4 deps pinned & verified against live PyPI in earlier sessions; Phase 5 adds no new dependency (reuses FlagEmbedding, still commented out); Phase 6 promotes httpx from dev-only to a real runtime dep (no new package, no version change)
│   ├── Dockerfile
│   └── .env / .env.example
├── frontend/
│   ├── src/
│   │   ├── App.tsx / main.tsx
│   │   ├── pages/Dashboard.tsx      # sidebar (upload+library) + chat placeholder panel
│   │   ├── components/UploadButton.tsx
│   │   ├── components/DocumentList.tsx
│   │   ├── hooks/useDocuments.ts
│   │   ├── services/api.ts          # fetch wrapper for all backend calls
│   │   └── types/document.ts
│   ├── vite.config.ts               # Tailwind v4 plugin + /api proxy to :8000
│   ├── Dockerfile
│   └── .env.example
├── data/{uploads,processed,samples}/
├── docker-compose.yml                # backend, frontend, postgres, qdrant (qdrant unused until Phase 4)
├── .env.example                      # root-level, source of truth for all env vars
├── PROJECT_STATE.md                  # <-- this file
└── README.md
```

---

## VERIFICATION LOG (Phase 1)

Run in the actual sandboxed environment, not just written speculatively:

- `pip install -r backend/requirements.txt` → succeeded.
- `pytest backend/tests/ -q` → **5 passed** (health check, reject unsupported
  type, reject empty file, full upload/list/get/delete roundtrip, path
  traversal filename sanitization).
- `uvicorn app.main:app` booted; `curl /api/health` → `200 OK`;
  `/docs` (Swagger UI) → `200 OK`.
- `npm install` + `npx tsc -b` → no type errors.
- `npm run build` → succeeded, Tailwind CSS compiled correctly (11.25 kB
  output CSS confirms utility classes were picked up, not just the
  unprocessed `@import`).
- **Bug found & fixed during a clean-room re-verification:** SQLite doesn't
  create its own parent directory, so a truly fresh checkout (no pre-existing
  `data/` folder) failed with `unable to open database file`. Fixed in
  `backend/app/database/session.py` by `mkdir(parents=True, exist_ok=True)`
  on the DB file's parent dir before the engine is created. Re-ran the full
  clean-room sequence afterward (fresh venv, fresh install, no `data/` dir
  present) — 5/5 tests passed, server booted, a real multipart file upload
  via `curl` returned a correct `DocumentOut` JSON response. This is the
  kind of bug that only shows up on a genuinely fresh clone, so if you hit
  a similar "works on my machine, fails in Docker/CI" issue later, check
  for directories that a library assumes already exist.

---

## VERIFICATION LOG (Phase 2)

Run in a sandboxed environment with live network access on 2026-09-26,
following the exact steps this file's earlier warning section prescribed:

- **Pin re-verified against live PyPI** (`pip index versions pymupdf`):
  the `1.26.4` pin written without network access was stale — latest is
  `1.28.2`. Updated `requirements.txt` accordingly and installed clean.
- `pip install -r backend/requirements.txt` → succeeded (all packages,
  including `pymupdf==1.28.2`, resolved and installed with no conflicts).
- `pytest backend/tests/ -q` → **first run: 2 of 13 failed.** Both were
  real bugs, found only by actually executing the code:
  1. `test_text_pdf_is_extracted_and_chunked` failed because the test
     fixture (`_make_text_pdf`) built pages with `page.insert_text()`,
     which draws one continuous unwrapped line — long paragraphs ran off
     the right edge of the page and were silently clipped, so PyMuPDF only
     extracted ~124 of the intended ~1900 characters, never triggering the
     multi-chunk code path at all. This was a test-fixture bug, not an
     extractor or chunker bug — confirmed by unit-testing `chunk_pages()`
     directly (that test already passed) and by reproducing the clipped
     extraction by hand outside pytest. Fixed by switching the fixture to
     `page.insert_textbox()` with an explicit wrapping rect.
  2. `test_image_upload_is_routed_to_ocr_holding_status` failed with
     `ValueError: non-hexadecimal number found in fromhex()` — the
     hardcoded "minimal 1x1 PNG" hex string had a typo and was not valid
     hex. Regenerated a real, correct 1x1 PNG's bytes with `zlib`/`struct`
     and replaced the literal.
  - After both fixes: **13 passed**, including a from-scratch clean-room
    run (`rm -rf data/` before running — same check Phase 1 already
    verified, re-confirmed here since Phase 2 touches the same DB
    bootstrap path).
- `uvicorn app.main:app` booted; `curl /api/health` → `200 OK`; `/docs` →
  `200 OK`. This time also verified the actual ingestion pipeline over a
  **real HTTP round trip** (not just Starlette's in-process TestClient):
  built a real 2-page PDF with PyMuPDF, `curl -X POST` it to
  `/api/documents`, then polled `/api/documents/{id}`:
  - Normal text PDF → `processing_status: completed`, `source_type:
    text_pdf`, `page_count: 2`, correct per-page `raw_text` at
    `/api/documents/{id}/pages`, and correct chunking at
    `/api/documents/{id}/chunks` (long page split into multiple
    page-scoped chunks with sequential `chunk_index`, short page as a
    single chunk).
  - Blank (no text layer) 2-page PDF → `processing_status: ocr`,
    `source_type: scanned_pdf`, `processing_error` correctly mentions
    Phase 3, no pages/chunks written.
  - Malformed `%PDF-1.4 not real` bytes → `processing_status: failed`
    with a clean, real PyMuPDF error message (`Failed to open file ...
    as type pdf`) — the background task did not crash the server.
- Frontend: `npm install`, `npx tsc -b` (no type errors), `npm run build`
  (succeeded, same 11.25 kB compiled Tailwind CSS as Phase 1 — Phase 2
  didn't touch the frontend, this just re-confirms nothing regressed).
- **Net result:** Phase 2 backend logic (`pdf_extractor.py`,
  `naive_chunker.py`, `ingestion/router.py`, `api/documents.py` changes)
  is correct as originally written and required no fixes. The two bugs
  found were both in the *test file*, not the implementation — but they
  would have masked a real regression in the chunker (the multi-chunk
  code path was never actually being exercised) if left unverified. This
  is exactly the kind of gap "written but not run" leaves open.

---

## VERIFICATION LOG (Phase 3)

Run in a sandboxed environment with live network access on 2026-09-26:

- **Pins verified against live PyPI:** `pip index versions paddlepaddle`
  (latest CPU wheel `3.3.1`, no CUDA/custom index needed — the
  `paddlepaddle.org.cn` custom index in this file's earlier "Verified
  Decisions" section is only required for GPU builds), `paddleocr` (`3.7.0`,
  satisfies the `[doc-parser]>=3.6.0` extras requirement from the original
  spec), `opencv-python-headless` (`5.0.0.93`). All three installed cleanly
  alongside the existing Phase 1/2 pins with no dependency conflicts.
- `from paddleocr import PaddleOCRVL` → imports cleanly. Package install and
  import are NOT the problem.
- **Real, reproduced network gap:** `PaddleOCRVL(pipeline_version="v1.6")`
  raises `Exception: No available model hosting platforms detected. Please
  check your network connection.` — traced into `paddlex`'s own
  `official_models.py`, which tries HuggingFace, ModelScope, AIStudio, and
  BOS in turn and gets a connection failure from all four in this sandbox
  (see the network-access warning under "CURRENT STATUS" above). This was
  reproduced directly (not assumed) both from a bare Python REPL and via the
  actual `process_document()` background-task path with `get_ocr_pipeline()`
  unmocked.
- Because of that, the pipeline object itself is mocked at the
  `get_ocr_pipeline()` boundary in `tests/test_phase3_ocr.py` — see that
  file's module docstring for exactly what's real vs. faked. Everything on
  the real side was genuinely executed: PyMuPDF page rasterization (real
  PDFs, real multi-page rendering, verified 1 PNG per page written with
  nonzero size), OpenCV preprocessing (real image in, real image out,
  verified it doesn't crash on a real 4x4 PNG or on genuinely corrupt image
  bytes), HTML table parsing (`_parse_html_table` — real regex-based
  parsing verified against real HTML table markup), block-label
  classification, and the full FastAPI background-task path (upload →
  detect → rasterize → preprocess → [faked] OCR call → parse results →
  write DocumentPage/DocumentElement/Chunk rows → chunk → `completed`).
- `pytest backend/tests/ -q` → **first run: 2 of 26 failed** — both were
  the two pre-existing Phase 2 tests that asserted the *old* Phase-2-era
  behavior (scanned/image uploads permanently stuck at an `"ocr"` holding
  status with a "Phase 3 not implemented" message). That behavior is
  intentionally superseded by Phase 3 — a scanned PDF or image now actually
  attempts OCR and reaches `"completed"` (mocked pipeline) or `"failed"`
  (real, unmocked pipeline in this sandbox) instead of stopping partway.
  Not a bug — updated both tests to monkeypatch a trivial fake pipeline and
  assert the document reaches `"completed"` with `processing_method:
  paddleocr_vl`, keeping their original job (confirming Phase 2's routing/
  detection is still correct) without duplicating Phase 3's own OCR-parsing
  tests.
- After that fix: **24 passed**, including a from-scratch clean-room run
  (`rm -rf data/` before running, same check Phase 1/2 already established).
- `uvicorn app.main:app` booted; `curl /api/health` → `200 OK`. Did **not**
  re-run a live `curl` round-trip against a real scanned PDF this session
  (unlike Phase 1/2's practice) because the one thing that would exercise —
  the real OCR call — is exactly the piece proven unreachable above; a live
  curl round-trip would just reproduce the same `OCRUnavailableError` path
  already covered by `test_ocr_unavailable_marks_document_failed_not_crash`.
- Frontend: not touched this session (Phase 3 is backend-only per the
  phase table) — not re-verified, no reason to expect regression.
- **Net result:** Phase 3's own code (parsing, routing, table extraction,
  chunking, DB writes, error handling, rasterization, preprocessing) is
  verified for real. The one thing NOT verified end-to-end against the
  actual PaddleOCR-VL model is the model call itself, and that gap is a
  sandbox network limitation, reproduced and documented rather than
  assumed. **The next session (in an environment with open network
  access) should run one real scanned PDF through the real pipeline before
  fully trusting OCR output quality/labels in production** — the parsing
  code depends on the documented `paddlex` block-label vocabulary
  (`_TABLE_LABELS`, `_FORMULA_LABELS`, etc. in `app/ocr/paddle_ocr.py`),
  which was confirmed by reading source, not by seeing real output.

---

## VERIFICATION LOG (Phase 4)

Run in a sandboxed environment on 2026-09-26:

- **Pins verified against live PyPI:** `qdrant-client` (latest `1.19.1`),
  `FlagEmbedding` (latest `1.4.2`), `numpy` (latest `2.5.3`, but pinned down
  to `2.3.5` instead — see next bullet).
- **Real dependency conflict found & fixed:** installing `numpy==2.5.3`
  broke the already-installed `paddlex` (a `paddleocr` dependency), which
  requires `numpy<2.4,>=1.24`. Pinned `numpy==2.3.5` (latest version
  satisfying both constraints) in `requirements.txt` instead.
- **Real, reproduced disk-space gap:** `pip install torch` alone (no
  `FlagEmbedding` even yet) failed with `OSError: [Errno 28] No space left
  on device`. Root cause: `torch`'s PyPI wheel bundles full NVIDIA CUDA
  libraries (`nvidia-cublas-cu12`, `triton`, etc.) totaling >10 GB; this
  sandbox had only a few GB free even after `pip cache purge` (which alone
  freed ~3.6 GB from a previous, unrelated install). Tried removing the
  installed NVIDIA/triton packages to make room — freed some space but not
  enough; a second `pip install torch` attempt still failed the same way.
  Also confirmed `download.pytorch.org` (the usual source for a much
  smaller CPU-only wheel) is not on this sandbox's network allow-list
  either, so that workaround isn't available here regardless of disk. This
  is a sandbox resource limitation, reproduced directly (not assumed).
- **Confirmed the real `FlagEmbedding` API shape without installing it:**
  `pip download FlagEmbedding==1.4.2 --no-deps -d <dir>` (small, no torch
  needed) then `unzip` + read
  `FlagEmbedding/inference/embedder/encoder_only/m3.py` directly. Confirmed:
  `from FlagEmbedding import BGEM3FlagModel` (aliases the real `M3Embedder`
  class), constructor signature `BGEM3FlagModel(model_name_or_path,
  use_fp16=bool, devices=str|list[str]|None, ...)` — note `devices`
  (plural), which `app/embeddings/embedder.py` was written against
  correctly from the start because of this check — and `.encode(texts,
  batch_size=..., return_dense=..., return_sparse=...,
  return_colbert_vecs=...)` returns a dict with keys `"dense_vecs"`
  (np.ndarray), `"lexical_weights"` (list[dict]), `"colbert_vecs"`
  (list[np.ndarray]).
- Because installing the real dependency wasn't even possible, the
  embedding model call is mocked at the `get_embedding_model()` boundary in
  `tests/test_phase4_embeddings.py` — see that file's module docstring.
  `get_embedding_model()` itself IS exercised for real against the genuine
  `ImportError` (`FlagEmbedding` truly isn't installed), confirming the
  boundary raises `EmbeddingUnavailableError` correctly without needing any
  mocking for that one test.
- **`qdrant-client`'s embedded/local mode needed no mocking at all** — real
  `QdrantClient(path=...)`, real `create_collection`/`collection_exists`,
  real `upsert`/`query_points`/`delete` with a real `FilterSelector`, all
  exercised directly in a scratch script before writing any test, then
  again for real inside `tests/test_phase4_embeddings.py`.
- **Real bug found while writing tests, fixed in `qdrant_store.py`'s
  docstring (not its code — the code was already correct):** local-mode
  Qdrant rejects a point id that isn't a valid UUID or unsigned int
  (`ValueError: Point id id-1 is not a valid UUID`) — the test fixtures
  originally used plain strings like `"id-1"`, which real Qdrant genuinely
  rejects; fixed the *test* fixtures to use real UUID strings (production
  code was never wrong here, since `Chunk.id` is already a UUID).
- `pytest backend/tests/ -q` → **first run: 5 of 43 failed** — three
  distinct, all found only by actually executing the code:
  1 & 2. The two invalid-point-id test bugs just described
  (`test_search_scoped_to_document_id`,
  `test_delete_document_vectors_removes_only_that_document`).
  3. `test_delete_document_cleans_up_qdrant_vectors` used an all-zero query
  vector, which is degenerate for cosine similarity — replaced with a
  proper unit vector.
  4 & 5. `test_phase2_ingestion.py::test_text_pdf_is_extracted_and_chunked`
  and (before an autouse-fixture fix) every Phase 3 OCR test that asserted
  `"completed"` failed, because Phase 4 changed what "completed" requires:
  a document now must also successfully embed+index before reaching
  `completed`, and the real embedding model genuinely isn't installed in
  this sandbox, so every pre-Phase-4 test doing a full pipeline run through
  the real (unmocked) `get_embedding_model()` now genuinely failed at that
  new step. This is the same kind of intentional supersession Phase 3 hit
  with Phase 2's OCR-holding-status tests — not a regression, a
  consequence of Phase 4 correctly changing the terminal status
  requirement. Fixed by adding an `autouse` fixture (`_fake_embedding_model`)
  to `test_phase2_ingestion.py` and `test_phase3_ocr.py` that fakes only
  the embedding call, keeping each file's own concern (extraction/chunking,
  OCR routing/parsing) genuinely tested without needing to know anything
  about Phase 4 beyond "return a vector of the right shape". Also updated
  `test_text_pdf_is_extracted_and_chunked`'s `embedded is False` assertion
  to `is True`, since chunks now genuinely get embedded in that test too.
- After all fixes: **43 passed**, including a from-scratch clean-room run
  (`rm -rf data/` before running).
- `uvicorn app.main:app` booted for real; `curl /api/health` → `200 OK`.
  **Also ran a real, unmocked HTTP round-trip** (not just TestClient) —
  uploaded a real single-page text PDF via `curl -X POST
  /api/documents`, and confirmed over real HTTP that: extraction and
  chunking succeeded for real (`page_count: 1`, one correct chunk written
  with the real extracted text), the document correctly stopped at
  `processing_status: "failed"` with `processing_error: "Embedding
  unavailable: The 'FlagEmbedding' package (and its 'torch' dependency) is
  not installed. ..."`, and the chunk row has `embedded: false` — the
  server did not crash and the failure is clean and diagnosable, exactly
  as designed.
- Frontend: not touched this session (Phase 4 is backend-only per the
  phase table) — not re-verified, no reason to expect regression.
- **Net result:** Phase 4's own code (embedding boundary, Qdrant
  collection/upsert/search/delete, DB bookkeeping, status transitions,
  error handling, and the shared `_embed_and_index_chunks()` wiring into
  both ingestion pipelines) is verified for real, including a genuine
  unmocked HTTP round-trip through the actual failure path. The one thing
  NOT verified end-to-end against the actual bge-m3 model is the model
  call itself, and that gap is a sandbox resource limitation (disk AND
  network), reproduced and documented rather than assumed. **The next
  session (in an environment with enough disk and open network) should
  install `FlagEmbedding`+`torch` for real, run one real document through
  the real pipeline, and sanity-check that embeddings actually come back
  1024-dimensional and roughly sensible (e.g. two similar sentences score
  higher than two unrelated ones) before fully trusting retrieval quality
  in production.**

---

## VERIFICATION LOG (Phase 5)

**Read this before trusting anything in Phase 5 — it is materially
different from every earlier phase's verification log.** Phases 1–4 each
installed real dependencies and ran the real test suite (`pytest`) in a
sandboxed session, with only specific external model weights faked at a
documented boundary. Phase 5 could not do that:

- `python3 -m venv` + `pip install qdrant-client` inside the fresh venv
  failed with `ERROR: Could not find a version that satisfies the
  requirement qdrant-client (from versions: none)` / `ERROR: No matching
  distribution found for qdrant-client` — this sandbox instance had **no
  outbound network access at all**, not even to PyPI, confirmed directly
  (not assumed) by this exact command. Since nothing could be installed —
  not FastAPI, not SQLAlchemy, not PyMuPDF, not `qdrant-client`, none of
  it — `pytest backend/tests/ -q` could not be run at all this session,
  including the 38 tests from Phases 1–4 that previously passed.
- Given that constraint, verification this session was done at whatever
  level was actually possible:
  1. **Syntax**: every new/modified `.py` file (`config.py`,
     `embedder.py`, `qdrant_store.py`, `fusion.py`, `reranker.py`,
     `hybrid.py`, `api/search.py`, `api/documents.py`, `main.py`,
     `ingestion/router.py`, `schemas/schemas.py`, and all 5 test files)
     compiled cleanly with `python3 -m py_compile` — catches typos/syntax
     errors, nothing about actual runtime behavior.
  2. **Real logic execution, dependency-free**: `app/retrieval/fusion.py`
     has zero external dependencies (pure Python), so it was imported and
     run directly against the same scenarios `test_phase5_retrieval.py`
     encodes (a chunk present in both the dense and sparse hit lists
     correctly outranks chunks present in only one; disjoint lists
     interleave by rank correctly) — this genuinely passed, output
     inspected by hand, not just "should work."
  3. **Real logic execution, with a hand-written local stub in place of
     `loguru`** (the one dependency `embedder.py` and `reranker.py` need
     that isn't otherwise available — a five-line stub module providing a
     no-op `logger.info/warning/error`, written purely to unblock this
     check, not part of the shipped project): `app/retrieval/reranker.py`
     (`rerank()`'s ordering/top_k/empty/crash-wrapping behavior, and
     `get_reranker_model()`'s genuine `ImportError` -> 
     `RerankerUnavailableError` boundary, since `FlagEmbedding` truly isn't
     installed here either) and `app/embeddings/embedder.py`'s new sparse
     path (`embed_texts()`'s sparse parsing including the length-mismatch
     fallback, `embed_query()`'s single-query unwrapping) were each
     exercised directly against hand-written fake models and confirmed to
     behave as the corresponding tests in `test_phase5_retrieval.py`
     assert.
  4. **NOT verified at all this session**: `app/retrieval/qdrant_store.py`
     (needs the real `qdrant-client` package — its named dense+sparse
     vector schema, `_to_sparse_vector()`'s point construction, and
     `search_dense`/`search_sparse` have never actually been run against a
     real Qdrant instance, embedded or otherwise), `app/retrieval/hybrid.py`
     (needs `qdrant_store.py`), `app/api/search.py` and the full
     `POST /api/search` HTTP path (needs FastAPI + the whole stack), and
     the updated Phase 2/3/4 test fixtures (needs the whole stack). These
     are the highest-value things for the next session with real network
     access to run first — see the numbered list under "KNOWN GAPS" below.
- **What this means concretely:** the qdrant-client API calls in this
  phase (`vectors_config={...}` / `sparse_vectors_config={...}` on
  `create_collection`, `PointStruct(vector={"dense": ..., "sparse":
  SparseVector(...)})`, `query_points(..., using="dense"/"sparse",
  query=...)`) are written to match qdrant-client's documented named-vector
  and sparse-vector API, consistent with the `qdrant-client==1.19.1`
  version already pinned in `requirements.txt` — but, unlike Phase 4's
  Qdrant code (which had zero mocking because the real client could
  actually run), none of it has been proven to actually work yet. Budget
  real time for this in the next session rather than assuming it's as
  solid as Phase 4's Qdrant integration was.

---

---

## VERIFICATION LOG (PHASE 6)

**This session's environment was, if anything, slightly more constrained
than Phase 5's — worth reading in full before trusting anything beyond
what's listed below.** Freshly re-confirmed this session (not assumed
carried-over from Phase 5's log):

- `python3 -m venv` + `pip install qdrant-client` inside a fresh venv
  failed identically to Phase 5: `ERROR: Could not find a version that
  satisfies the requirement qdrant-client (from versions: none)` — no
  outbound network access at all, PyPI included.
- Separately, and new information this session: the sandbox's system
  Python (no venv at all) has **none** of `requirements.txt` importable
  — `import httpx`, `import fastapi`, `import pydantic`, `import
  sqlalchemy`, `import loguru`, `import pydantic_settings`, `import
  pytest` all raised `ModuleNotFoundError` directly, one by one. Phase 5's
  log didn't check this explicitly (it inferred "nothing installable"
  from the failed `pip install`); this session confirmed it two ways.
- Given that, verification this session was done at whatever level was
  actually possible, same philosophy as Phase 5:
  1. **Syntax**: every new file (`app/generation/prompt.py`,
     `app/generation/citation.py`, `app/generation/llm_client.py`,
     `app/generation/generator.py`, `app/api/chat.py`,
     `tests/test_phase6_generation.py`) and every modified file
     (`app/config.py`, `app/schemas/schemas.py`, `app/main.py`) compiled
     cleanly with `python3 -m py_compile`.
  2. **Real logic execution, dependency-free — the genuine win this
     session**: `app/generation/citation.py` and `app/generation/prompt.py`
     import nothing beyond the standard library (`re`, `dataclasses`,
     `typing`), so both were imported directly and exercised against real
     scenarios with a plain `python3` script (no pytest needed, since
     pytest isn't installed either) — output inspected by hand, not
     assumed:
     - a fully-cited two-sentence answer citing valid passage numbers ->
       `fully_cited=True`, zero invalid/uncited
     - an answer citing passage `[7]` when only 3 passages existed ->
       `invalid_indices == [7]`, `fully_cited=False`
     - a two-sentence answer where only the second sentence has a
       citation -> `uncited_sentence_count == 1`, `fully_cited=False`
     - an `INSUFFICIENT_EVIDENCE:`-prefixed answer -> short-circuits to
       `is_insufficient_evidence=True`, `fully_cited=True`, no citations
       parsed at all
     - prefix matching is case-insensitive and requires a true prefix,
       not a substring match anywhere in the text
     - zero passages + an uncited plain sentence -> correctly flagged as
       not fully cited (not a false-positive pass just because
       `num_passages == 0`)
     - `to_context_passages()` correctly 1-indexes and preserves input
       order against a two-item fake-result list; `build_context_block()`
       renders `[n] (source: file, page N)` correctly and returns the
       literal string `"(no passages retrieved)"` for an empty list;
       `build_user_prompt()` embeds both the question and every `[n]`
       marker; `SYSTEM_PROMPT` contains the `INSUFFICIENT_EVIDENCE`
       literal it's supposed to instruct.
     All of the above genuinely ran and passed — this is real
     verification, not a description of intended behavior. The exact
     script is reproducible from the scenarios listed in
     `tests/test_phase6_generation.py`, which mirrors it as real pytest
     tests for whenever pytest is installed.
  3. **NOT verified at all this session**: `app/generation/llm_client.py`
     (needs `httpx`, not installed — not even a `MockTransport`-based test
     could be attempted), `app/generation/generator.py` (needs the full
     Phase 4/5 import chain — `app.config` alone needs
     `pydantic_settings`), `app/api/chat.py` and the full `POST /api/chat`
     HTTP path (needs FastAPI + the whole stack). A hand-written
     `loguru`+`qdrant_client` stub pair (same trick Phase 5 used for
     `loguru` alone, to isolate `reranker.py`/`embedder.py`) was
     attempted for `generator.py` specifically, but abandoned: unlike
     Phase 5's reranker/embedder isolation, `generator.py`'s import chain
     runs through `app.config` (needs real `pydantic_settings`) and
     `app.retrieval.qdrant_store` (needs `qdrant_client.http.models`,
     not just `qdrant_client` itself) — stubbing that whole surface
     convincingly was judged not worth the effort versus just being
     honest that it's unverified and needs a real environment. These are
     the highest-value things for the next session with real network
     access to run first — see "KNOWN GAPS" below.
- **What this means concretely**: Phase 6 ships with its two purely-
  algorithmic pieces (citation validation, prompt construction) genuinely
  battle-tested against real scenarios — probably the strongest
  verification confidence of any single Phase 5/6 module so far, since it
  didn't just avoid crashing, it produced the exact expected structured
  output on every scenario tried. The two pieces that talk to something
  external (`llm_client.py`'s HTTP calls) or something with a heavy
  dependency chain (`generator.py`, `api/chat.py`) are unverified and
  should be treated with the same caution as Phase 5's untested
  `qdrant_store.py`/`hybrid.py`/`api/search.py` — written carefully and
  consistently with the rest of the codebase's patterns, but not proven
  to work yet.

---

## VERIFICATION LOG (Phase 11)

No Docker daemon available (`docker`/`docker-compose`: command not found,
confirmed this session). What WAS actually run/verified in this sandbox:

- `python -c "import yaml; yaml.safe_load(open('docker-compose.yml'))"` ->
  parses cleanly both before and after this phase's edits; a follow-up
  script asserted the two new keys (`backend.environment.QDRANT_MODE ==
  "server"`, `frontend.build.args.VITE_API_BASE_URL ==
  "http://localhost:8000"`) round-trip correctly through the YAML parser.
- `pip index versions psycopg2-binary` against live PyPI -> confirmed
  `2.9.13` exists and is latest; then `pip install psycopg2-binary==2.9.13`
  followed by `python -c "import psycopg2"` -> both succeeded in this
  sandbox's venv, confirming the binary wheel installs with no system
  Postgres headers/dev libs needed (which matters since the *backend*
  Docker image's base is `python:3.12-slim`, which also has none of those
  installed).
- `pytest backend/tests/ -q` re-run after the `requirements.txt` change ->
  still `119 passed`, confirming adding `psycopg2-binary` didn't disturb
  anything already working (it's additive; nothing was upgraded/removed).
- What could NOT be verified here, at all: whether the `backend`/
  `frontend` images actually build (no `docker build`), whether the exact
  `apt-get install` package names resolve on the actual `python:3.12-slim`
  base image's Debian release, whether nginx actually serves the built
  frontend correctly, and whether the full multi-container stack (Postgres
  + Qdrant server mode + backend + frontend) actually reaches a healthy
  state together. All of this is now explicitly the first thing item 2
  below asks for once an environment with a real daemon is available.

## VERIFICATION LOG (Phase 10)

Commands actually run this session, in this sandbox, with real output:

- `python -m venv` + `pip install -r backend/requirements.txt` (already-
  satisfied from Phase 7's environment work) then
  `pip install pytest pytest-asyncio` then `pytest backend/tests/ -q` ->
  `119 passed, 1 warning in 21.94s`. The one warning is a pre-existing
  `anyio`/`starlette` `DeprecationWarning`, unrelated to this phase's
  changes.
- Immediately after that run: `ls backend/data/` showed only the
  pre-existing `processed/` and `uploads/` directories from earlier manual
  runs — no `dev.db` and no `qdrant/` — confirming `conftest.py`'s
  temp-path redirection actually took effect and the suite no longer
  touches the real dev DB/vector store.
- `npm install` (added `vitest`, `@testing-library/react`,
  `@testing-library/jest-dom`, `jsdom` to `frontend/`) -> clean install, no
  errors.
- `npx vitest run` -> `Test Files  2 passed (2)`, `Tests  9 passed (9)`.
- `npx tsc -b` -> exit 0, no output (clean). `npm run build` -> clean
  production build (`vite build` succeeded, same as Phase 9's verified
  build, confirming the new devDependencies/config didn't regress it).

## KNOWN GAPS / THINGS THE NEXT SESSION SHOULD DO FIRST

1. Phase 1 tests share one SQLite file across test runs (`data/dev.db`) —
   fine for now, but Phase 10 (proper test suite) should switch tests to an
   in-memory SQLite DB or a temp file per test session. Phase 4 tests hit
   the same pattern for the local Qdrant collection at `data/qdrant/` —
   same fix should cover both when Phase 10 gets to it.
2. `docker-compose.yml` has NOT been run end-to-end yet (no `docker compose up`
   performed in this sandbox, still — no Docker daemon available in any
   session including Phase 11's). Phase 11 did a full static review instead
   and found/fixed four real bugs that would otherwise have surfaced on
   first real run (missing `QDRANT_MODE=server`, missing `psycopg2-binary`,
   missing OS libs for OpenCV/Paddle in `backend/Dockerfile`, and no way
   for the built frontend to reach the backend at all) — see "CURRENT
   STATUS" above for the full list. Verify the actual `docker compose up`
   once an environment with a daemon is reached — this now also needs to
   verify `QDRANT_MODE=server` against the real `qdrant` service, not just
   the `local` mode Phase 4 verified here, AND double-check Phase 11's four
   fixes actually work as reasoned (they're confident-by-inspection, not
   run-verified).
3. ~~`frontend/.gitignore` and root `.gitignore` should be checked...~~ —
   **partly done (Phase 11):** root `.gitignore` had a real bug (anchored
   `data/uploads/*`/`data/processed/*` patterns that missed
   `backend/data/...` entirely, and no `data/qdrant/` pattern at all) —
   fixed as part of Phase 11's Docker review since it's the same
   docker-data surface. `frontend/.gitignore` was reviewed and is already
   correct (`node_modules`, `dist`, etc. all present). Still worth one more
   pass before actually pushing to a real git host (Phase 12), just to
   catch anything project-specific this review might have missed.
4. `data/processed/{document_id}/pages/*.png` (Phase 3's rasterized scanned-
   PDF pages) accumulate on disk with no cleanup/retention policy yet —
   fine for a dev/demo build, but worth a TTL or explicit cleanup step
   before anything resembling production use.
5. Neither the real PaddleOCR-VL model (Phase 3) nor the real bge-m3
   embedding model (Phase 4) has ever actually been run against a real
   document in any session so far — both network-blocked (OCR) or
   network+disk-blocked (embeddings) in this sandbox. First thing to do
   with an environment that has both open network access AND enough disk:
   install both sets of real deps, run one real scanned PDF and one real
   text PDF through `POST /api/documents` with neither `get_ocr_pipeline()`
   nor `get_embedding_model()` mocked, and sanity-check both the OCR block
   labels (`app/ocr/paddle_ocr.py`'s `_classify_block()`) and the embedding
   output shape/quality against what the code currently assumes.
6. ~~**Do this FIRST, before any Phase 7 code:** get a real Python
   environment...~~ — **DONE this session (Phase 7).** `pip install -r
   backend/requirements.txt` succeeded completely; `pytest backend/tests/
   -q` passed 84/84 (now 102/102 with Phase 7's own new tests), covering
   everything this item asked for: `test_phase5_retrieval.py`'s
   `qdrant_store.py` named vectors/`hybrid.py`/`/api/search`, and
   `test_phase6_generation.py`'s `generator.py`/`/api/chat`. Part (c) —
   `pip download FlagEmbedding --no-deps` to confirm `reranker.py`'s
   `FlagAutoReranker` API shape by reading source — was NOT done this
   session (not blocking Phase 7; still worth doing, folded into item 8
   below since it needs the same network+disk environment anyway).
7. Phase 5 reused the same bge-m3 model/call for sparse retrieval as
   planned (`return_sparse=True` in `embed_texts()`, threaded through
   `EmbeddingResult.sparse`) — done, see the Phase 5 design section above.
8. Whatever environment finally has both open network access and enough
   disk to install `FlagEmbedding`+`torch` for real (still not this
   session's sandbox — see item 5 above, unchanged from Phase 4) should,
   in one pass: install both, run one real scanned PDF through OCR and one
   real document through embedding+indexing+hybrid search+reranking+
   generation end to end (now including a real LLM call, Phase 6 permitting
   — see item 9 below for the API-key prerequisite), and sanity-check all
   of it together — OCR block labels, embedding shape/quality, real
   dense+sparse Qdrant search, real reranker output, AND now a real
   generated answer with real citation validation — since none of the
   last four have ever actually run against real weights/APIs in any
   session so far.
9. **New (Phase 6): an actual `LLM_API_KEY` for whichever `LLM_PROVIDER`
   is tested first.** Nothing in this repo can supply one — the next
   session (human or Claude) needs a real DeepSeek/OpenAI/Gemini API key
   in `.env` before `/api/chat` can do anything beyond hit the
   `LLMUnavailableError` -> 503 path. Consider testing all three providers
   at least once each (switch `LLM_PROVIDER`+`LLM_MODEL`+`LLM_API_KEY` and
   re-run one `/api/chat` call) since `llm_client.py`'s three code paths
   have equally never been run.
10. **New (Phase 6): once `/api/chat` is confirmed working end to end,
    sanity-check the citation-validation output against the project's
    demo question set** (see "ORIGINAL REQUIREMENTS REFERENCE" below) —
    specifically, confirm real LLM output actually follows the `[n]`
    citation convention and the `INSUFFICIENT_EVIDENCE:` prefix the way
    `SYSTEM_PROMPT` asks it to. `citation.py`'s parsing logic is verified
    correct against hand-written fake answers (see Verification Log
    (Phase 6)) but has never seen a real LLM's actual output style, which
    may deviate from the prompt's instructions in ways worth tightening
    the prompt or the parser for (e.g. a model that cites as "(passage 2)"
    instead of "[2]," or forgets the exact `INSUFFICIENT_EVIDENCE:`
    casing/prefix).
11. **New (Phase 7): confidence/conflict thresholds are documented
    heuristics, not calibrated.** `confidence.py`'s 50/50 weighting and
    0.66/0.33 level thresholds, and `conflict.py`'s 0.5 similarity
    threshold / 5% relative tolerance, were reasoned about carefully but
    never tuned against real labeled examples (no such dataset exists in
    this project). Once Phase 7 has seen real LLM answers (needs item 9's
    API key) and, ideally, some real multi-document corpora with actual
    known contradictions, revisit these constants — the module docstrings
    explain the reasoning behind each current value if this comes up.
12. ~~**Phase 8 (next)**: `app/conversations/` is currently an empty
    package...~~ — **DONE this session (Phase 8).** See "CURRENT STATUS"
    above for the full design/implementation summary. One item genuinely
    still open from it: `resolve_followup()`'s rewrite call has never run
    against a real LLM (same unverified-HTTP-boundary situation as
    `llm_client.py`'s generation call since Phase 6 — see item 9's
    original note) — once a real `LLM_API_KEY` is available, sanity-check
    that a real model's rewrite output is actually a bare standalone
    question with no preamble the way `REWRITE_SYSTEM_PROMPT` asks for,
    the same way item 10 asks for the generation prompt/citation format.
13. ~~**Phase 9 (next)**: `frontend/` is still the kickoff scaffold...~~ —
    **DONE this session (Phase 9).** See "CURRENT STATUS" above for the
    full design/implementation summary and its "Known Phase 9 gaps"
    subsection for what's genuinely still unverified (no browser in this
    sandbox to click through a real answer; `/api/chat` itself is still
    blocked on the same pre-existing FlagEmbedding/torch/API-key gaps).
14. ~~**Phase 10 (next)**: proper test suite hardening...~~ — **DONE this
    session (Phase 10).** See "CURRENT STATUS" above for the full summary.
    (a) fixed via `backend/tests/conftest.py` (session-scoped temp
    DB/Qdrant path/upload dirs, set before any `app.*` import). (b) fixed
    via `vitest` + React Testing Library, with `src/services/api.test.ts`
    and `src/hooks/useChat.test.ts` (9 tests, all passing). Still open:
    no component-level tests (`ChatMessage`/`EvidencePanel`/`DocumentList`)
    — a reasonable Phase 11+ increment if frontend regressions become a
    real concern, but not one of the two gaps this phase targeted.
15. **New (Phase 10) — nice-to-have, not blocking:** consider a small CI
    config (GitHub Actions or similar) that runs `pytest backend/tests/`
    and `npm run test`/`tsc -b`/`npm run build` on every push, now that
    both suites are fast and isolated enough to run unattended. No such
    config exists yet — this was out of scope for Phase 10 itself (no CI
    provider/repo host access in this sandbox to verify a workflow file
    actually runs), but is worth doing whenever this project moves to a
    real git host.

---

## ORIGINAL REQUIREMENTS REFERENCE

The full original project brief (all 8 key requirements, full tech stack
mandate, PaddleOCR-VL-1.6 requirement, demo question set, success criteria,
coding rules, etc.) was provided as a single large spec at project kickoff.
If a future session needs the full verbatim spec and it isn't otherwise
available, ask the user to re-paste it — do not reconstruct it from memory.
