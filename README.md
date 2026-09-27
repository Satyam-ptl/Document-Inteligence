# Document Intelligence & Multi-Source Search Platform

> **Status (updated Phase 12): Phases 1–12 implemented. Backend logic —
> ingestion, chunking, OCR block-labeling, hybrid retrieval fusion,
> citation validation, confidence/conflict detection, conversation
> follow-up resolution — is REAL-VERIFIED via `pytest backend/tests/ -q`
> (125/125 passing) with a 90% coverage gate in an environment with a full Python install. The
> frontend build (`tsc -b` + `vite build`) and its own test suite
> (`vitest`, 9/9 passing) are likewise real-verified. What is NOT yet
> verified, because no sandbox session so far has had the resources for
> it: the real PaddleOCR-VL and bge-m3/bge-reranker model weights have
> never actually run (network/disk-blocked), no real LLM API call has ever
> been made (no API key available in this environment), and
> `docker compose up` has not been executed in this environment (no Docker daemon
> available). Production configuration now fails fast unless PostgreSQL,
> server-mode Qdrant, an API key, HTTPS CORS origins, an external LLM, and
> real embeddings are configured. The compose stack still needs one run in
> an environment with Docker and model/API credentials. See
> [`PROJECT_STATE.md`](./PROJECT_STATE.md)'s
> "KNOWN GAPS" section for the exact, prioritized list of what a future
> session with those resources should do first.**
> See [`PROJECT_STATE.md`](./PROJECT_STATE.md) for exactly what's built,
> what's verified, and how to resume this project in a new session if
> needed. This README is filled in incrementally as phases complete —
> All planned application phases are implemented; the remaining caveats below
> are environment-dependent verification items, not unbuilt features.

## 1. Problem Statement

Organizations manage large collections of PDFs, scanned documents, reports,
manuals, tables, and images. This platform ingests heterogeneous documents,
routes each through the right processing pipeline, and answers
natural-language questions by retrieving and combining evidence from
multiple sources — with full source attribution and no fabricated answers.

## 2. Architecture

```
Document Upload
      ↓
Document Type Detection        (Phase 2/3)
      ↓
Multi-Pipeline Processing      (PyMuPDF | PaddleOCR-VL-1.6 | python-docx | table parser)
      ↓
OCR / Structure Extraction     (Phase 3 — verified; real model weights unreachable in this sandbox)
      ↓
Chunking + Metadata            (Phase 2/3)
      ↓
Embeddings                     (Phase 4)
      ↓
Hybrid Semantic Retrieval      (Phase 5 — dense + sparse + fusion)
      ↓
Reranking                      (Phase 5)
      ↓
Multi-Source Context           (Phase 5/6)
      ↓
AI-Powered Response            (Phase 6 — grounded, no fabrication)
      ↓
Evidence + Source Attribution  (Phase 6/9)
```

### Intelligent document router (Phase 2/3)

```
                     DOCUMENT ROUTER
                             |
             +---------------+----------------+
             |               |                |
         TEXT PDF       SCANNED PDF        IMAGE
             |               |                |
         PyMuPDF       PaddleOCR-VL      PaddleOCR-VL
```

PDF routing currently uses the document's average extracted text to choose
the text-PDF or scanned-PDF pipeline. Mixed text/image pages are handled by
the selected pipeline; page-level mixed-mode routing is not currently
implemented.

## 3. Technology Stack

| Layer | Choice | Status |
|---|---|---|
| Frontend | React + Vite + TypeScript + Tailwind CSS v4 | ✅ Phase 1 |
| Backend | FastAPI + Pydantic + Uvicorn | ✅ Phase 1 |
| Database | SQLite (dev) / PostgreSQL (prod), via `DATABASE_URL` | ✅ Phase 1 |
| PDF parsing | PyMuPDF | ✅ Phase 2 |
| OCR / document understanding | **PaddleOCR-VL-1.6** | ✅ Phase 3 (routing/parsing code verified; real model weights unreachable in this sandbox — see `PROJECT_STATE.md`) |
| Image preprocessing | OpenCV | ✅ Phase 3 |
| DOCX / XLSX / CSV | python-docx / openpyxl / table parser | ✅ Supported |
| Vector DB | Qdrant | ✅ Phase 4/5 (local mode tested; server mode needs deployment verification) |
| Embeddings | Configurable (default `BAAI/bge-m3`) | ✅ Phase 4 (hashing fallback tested; real model weights need environment verification) |
| Reranker | Configurable (default `BAAI/bge-reranker-v2-m3`) | ✅ Phase 5 (fallback behavior tested; real weights need environment verification) |
| LLM | Configurable (DeepSeek / Gemini / OpenAI via env var, raw REST) | ✅ Phase 6 (clients and failure paths tested; live provider call requires credentials) |
| Containerization | Docker + docker-compose | ✅ Phase 11/12 (configuration validated; full runtime needs a Docker daemon) |

## 4. Project Structure

See `PROJECT_STATE.md` → "What exists right now" for the current, accurate
file inventory (kept up to date per phase).

## 5. Installation & Running Locally (Phase 1 — works today)

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp ../.env.example .env          # edit if needed; SQLite works out of the box
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for the interactive API docs, or
http://localhost:8000/api/health.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173. The dev server proxies `/api` requests to
`http://localhost:8000`, so run the backend first.

### Tests

```bash
cd backend
pytest tests/ -q --cov=app --cov-report=term-missing --cov-fail-under=70
```

The current backend suite has **125 passing tests** and enforces a **70%
minimum coverage gate** (the latest run reached approximately 90% coverage).
OCR and external model boundaries use deterministic fakes in tests; live
model/provider verification requires the corresponding network access,
credentials, and model weights.

### GitHub setup

From the repository root:

```bash
git init
git add -A
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

The repository includes a root `.gitignore`, `.gitattributes`, and GitHub
Actions workflow at `.github/workflows/ci.yml`. Generated data, secrets,
virtual environments, `node_modules`, and frontend build output are ignored.
Only example environment files should be committed.

The base backend install is complete for the supported local fallback:

```bash
cd backend
python -m pip install -r requirements.txt
```

For real BGE-M3 embeddings and reranking, install the optional heavyweight
stack as well:

```bash
python -m pip install -r ../requirements-ml.txt
```

## 6. Environment Variables

See [`.env.example`](./.env.example) for the full list with comments. Never
commit a real `.env` file — copy it and fill in secrets locally.

## 7. Docker (Phase 11/12 — configuration validated; runtime verification required)

```bash
docker compose up --build
```

## 8. Demo Questions

1. "What was the company's revenue in 2025?"
2. "Compare revenue between 2024 and 2025."
3. "What was the reason for the revenue increase?"
4. "What about 2023?" *(follow-up context test)*
5. "What is the total amount on invoice INV-2026-0187?"
6. "What does the employee leave policy say about casual leave?"
7. "What is the highest revenue year in the table?"
8. "Calculate the percentage increase in revenue from 2024 to 2025."
9. "What was the company's revenue in 2035?" *(expected: insufficient evidence)*
10. "Why do the two reports show different revenue figures?" *(expected: conflict identification)*

## 9. Reliability Strategy (implemented starting Phase 7)

- Never fabricate facts or citations — every citation is validated against
  actually-retrieved chunk metadata before being shown.
- Explicit "insufficient evidence" response when retrieval confidence is low.
- Conflicting sources are surfaced side-by-side, never silently resolved.
- Calculations (growth %, sums) are computed programmatically, not by the LLM.

## 10. Limitations (current)

- Normal (text-layer) PDFs are ingested and chunked (Phase 2) — verified.
- Scanned PDFs and images are routed through PaddleOCR-VL-1.6, producing
  structured layout elements (paragraphs, headings, tables, formulas,
  images) and page-scoped chunks with tables kept whole (Phase 3) —
  verified, **except** the actual OCR model call, which can't reach its
  model weights in this build sandbox's network allow-list. See
  `PROJECT_STATE.md` for the exact reproduced error and what to do first
  with open network access.
- `.xlsx` and `.csv` uploads are extracted as searchable table chunks.
  `.docx` uploads are extracted with `python-docx`, including paragraphs and tables.
- Chat, retrieval, citation validation, confidence scoring, conflict detection,
  and conversation follow-up handling are implemented and covered by tests.
- The Docker Compose configuration is validated with `docker compose config`,
  but the full stack has not been run in this environment because no Docker
  daemon is available.

## 11. Roadmap

See the phase table in `PROJECT_STATE.md`.

## 12. License

This project is licensed under the [MIT License](./LICENSE).
