# MyLittleLawyer FastAPI (GenAI Architecture Skeleton)

This repository provides a clean, scalable GenAI project structure aligned with the shared architecture. It includes placeholders only (no implementation code) for LLM abstraction, RAG, preprocessing, inference, and testing. The stack is designed for:

- LangGraph for orchestration
- Gemini API as the model provider
- Pinecone as the vector database
- FastAPI as the backend service

## Project Structure

Key top-level directories:

- `config/`: configuration files for models and logging
- `data/`: cache, embeddings, and vector store data placeholders
- `src/`: application modules (core, prompts, rag, processing, inference)
- `tests/`: unit and integration test placeholders
- `scripts/`: automation placeholders

## Environment Variables

- Gemini API keys and model name
- Pinecone API keys, index name, and host
- Document source paths/URLs for legal content ingestion

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The API container exposes port `8000` and runs:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

