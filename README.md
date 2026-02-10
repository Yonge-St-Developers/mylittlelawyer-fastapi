# Lahwita FastAPI (GenAI Architecture Skeleton)

This repository contains a GenAI stack for Lahwita (Ontario LTB assistant), with working scaffolding for RAG, graph orchestration, and document processing. It is wired for:

- LangGraph orchestration
- Gemini API (text + embeddings)
- Pinecone for legal sources
- ChromaDB for CanLII case retrieval
- FastAPI backend + Streamlit client

## Services And Facilities

Services (HTTP endpoints):

- `GET /health`: health check.
- `POST /ai/chat`: main form chat endpoint (discovery, Q&A, form fill, confirmation).
- `POST /ai/file`: PDF generation handoff (stubbed response).
- `POST /lahwita/ai/indexr/`: index configured document sources into Pinecone.
- `POST /ai/consultant`: consultant mode powered by CanLII case retrieval.

Facilities (platform capabilities):

- Dual RAG pipelines: Pinecone for LTB rules/forms and ChromaDB for CanLII cases.
- CanLII crawler + local PDF indexer for case ingestion.
- LangGraph graphs for chat, file generation, and consultant flows.
- Document processing utilities (PDF-to-text, chunking, preprocessing).
- Streamlit UI for Form + PDF and Consultant modes.
- Unit and integration tests in `tests/`.

## Modes

User-facing modes (Streamlit UI):

- `Form + PDF`: multi-step form workflow (discovery → Q&A → fill → confirm) and PDF generation.
- `Consultant`: legal consultant chat using CanLII case retrieval and LTB rules.

Internal chat intents (used to pick prompts in `/ai/chat`):

- `describing_situation`: user describes their scenario (form discovery).
- `asking_about_form`: user asks questions about a known form (Q&A).
- `answering_form_field`: user provides field answers (form fill).
- `other`: fallback/confirmation.

## CanLII Usage (2025–2026 Cases)

The consultant pipeline is designed to use:

- LTB rules/instructions from Pinecone indexes.
- Official CanLII LTB cases from 2025 and 2026.

Case ingestion options:

- Crawl CanLII pages (default start URL is the 2026 LTB date index).
- Index local PDFs from `data/cache/ltb_law_case` without crawling.

The Streamlit consultant mode exposes a `Refresh CanLII index before answering` toggle to re-index before retrieval.

## Environment Variables

Key environment variables (see `.env.example` for the full list):

- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`
- Pinecone: `PINECONE_API_KEY`, `PINECONE_HOST`, `PINECONE_CLOUD`, `PINECONE_REGION`
- Indexes: `INDEX_*` for LTB rules, forms, and instructions
- ChromaDB: `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION`
- Document sources: `DOC_*` for ingestion paths/URLs

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The API container exposes port `8000` and runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

See `STRUCTURE.md` for the complete file map.
