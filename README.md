# Turiy.chat – UPS Assignment Submission

## Overview
Turiy.chat is a Retrieval-Augmented Generation (RAG) application that ingests PDF reports, builds hybrid vector/sparse indexes, and serves a NotebookLM-inspired chat UI. The system supports FAQ fast lane answers, hybrid retrieval, table-to-graph visualization, and conversation logging.

Key pieces:
- **Frontend**: Vanilla HTML/CSS/JS (`frontend/`) with three-panel layout, document manager, chat panel, sources pane, markdown rendering, and chart display.
- **Backend**: FastAPI (`backend/main.py`) exposing upload, rebuild, and query endpoints that delegate to a modular engine.
- **Engine modules**:
  - `ingestion/`: document loaders, table extraction, LLM refinement.
  - `vector_store/`: FAISS dense index + BM25 sparse index.
  - `retrieval/`: hybrid retriever with reciprocal rank fusion.
  - `orchestration/`: tool router, FAQ search, table-to-graph tool, answer synthesis.
  - `storage.py`: metadata persistence via SQLite or Supabase, plus conversation logging.
- **LLMs & Embeddings**: Gemini 2.5 Pro/Flash for routing/synthesis, GPT-4 optional fallback, HuggingFace Nomic embeddings for vector search.

## Prerequisites
- Python 3.10+
- Node tooling not required (frontend is vanilla)
- Optional: Supabase project (URL, service role key, Postgres URL)
- Required API keys (set in `backend/.env`):
  - `GEMINI_API_KEY`
  - `OPENAI_API_KEY` (optional)
  - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_DB_URL` if Supabase enabled

## Installation
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Ensure .env contains required keys
```

Frontend uses Tailwind CDN; no build step. Serve backend and open `frontend/index.html` in browser (or serve statically via any HTTP server).

## Running the Backend
```bash
cd backend
venv\Scripts\activate
python main.py
# FastAPI starts at http://127.0.0.1:8000
```

During startup:
- FAQ cache loads from `backend/data/faq.json`.
- Vector stores initialize (FAISS/BM25) for any configured modes.
- Metadata store (Supabase or SQLite) becomes available.

## Frontend Usage
1. Open `frontend/index.html` in your browser (Double-click or serve via `python -m http.server`).
2. Upload PDFs using the “Upload Documents” button (accepts `.pdf`).
3. Trigger “Re-Build Index” to re-ingest documents (see rebuild options below).
4. Enter queries in chat pane:
   - Toggle between `turiy-standard` and `turiy-thinking` modes.
   - Use regenerate button to escalate to thinking mode.
   - Sources pane lists citations; charts embed inline when `table_to_graph` fires.
5. Selected documents (checkboxes) restrict retrieval to chosen files.

### Managing FAQ Fast-Lane
- FAQ responses are stored in `backend/data/faq.json`. Create this file manually if it does not exist.
- The JSON should map normalized question strings to answer text, e.g.:
  ```json
  {
    "what is this app": "I am the 'GRI-Agent'..."
  }
  ```
- After editing `faq.json`, restart the backend (or call `engine.ensure_faq_initialized()` via a REPL) so the cache reloads. Keep FAQs concise and citation-ready.

## Rebuilding Indexes
Full rebuild (standard mode):
```bash
curl -X POST "http://127.0.0.1:8000/rebuild-index" \
  -H "Content-Type: application/json" \
  -d "{\"api\":\"google\",\"mode\":\"turiya-standard\"}"
```

Thinking mode rebuild (LLM-based chunking):
```bash
curl -X POST "http://127.0.0.1:8000/rebuild-index" \
  -H "Content-Type: application/json" \
  -d "{\"api\":\"google\",\"mode\":\"turiya-thinking\"}"
```

Notes:
- Rebuild currently processes **all** PDFs in `backend/data/` (no incremental ingest yet).
- Thinking mode requires Gemini API access; expect longer runtimes and occasional retries if API throttled.

## Conversation Logging
- Each query/response pair is logged through `storage.py` into `conversations` table (SQLite or Supabase).
- Captured fields: timestamp, API, mode, selected tool, latency, user query, answer text, sources metadata, chart metadata, prior chat turns, selected documents.
- Use these logs for analytics, memory replay, or debugging.

## Project Structure
```
backend/
  main.py               # FastAPI entrypoint
  engine.py             # Core orchestration for ingestion/retrieval/query
  ingestion/            # Document parsing & chunking modules
  vector_store/         # FAISS + BM25 helpers
  retrieval/            # Hybrid retriever implementation
  orchestration/        # Tool router, graph tool, synthesis
  config/agent_config.yaml
  data/                 # PDFs, vector store artifacts, FAQ
  storage.py            # Metadata stores (SQLite/Supabase) + conversation logging
frontend/
  index.html            # UI layout with Tailwind CDN
  style.css             # Custom styles (scrollbars, collapsed panes)
  app.js                # Frontend logic (fetch, chat, rendering)
bugs.md                 # Known issues list
update_features.md      # Planned updates & enhancements
new_features.md         # Future feature ideas
turiy_architecture.md   # Architecture diagram overview
```

## Development Tips
- Run backend with `reload=False` to avoid file locks on Windows during FAISS rebuilds.
- Delete `backend/data/vector_store/*` directories manually if you need a clean slate.
- Linter warnings can be checked via `task lint` (if defined) or `pylint`/`flake8`.
- Use Supabase SQL editor to inspect `documents`, `chunks`, and `conversations` tables.

## Known Limitations
- No partial rebuilds; full re-ingestion required after each upload (tracked in `update_features.md`).
- Gemini router may occasionally misroute; heuristics implemented but monitoring ongoing (`bugs.md`).
- Streaming responses and prompt customization not yet exposed (see `new_features.md`).

## License / Attribution
This submission is provided as part of the UPS assignment. External libraries retain their respective licenses (FastAPI, LangChain, PyMuPDF, Matplotlib, etc.). Configure API keys responsibly and review provider terms (Google Generative AI, OpenAI, Supabase) before deployment.

