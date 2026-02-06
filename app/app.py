from fastapi import FastAPI

from app.schemas import (
    ChatRequest,
    ChatResponse,
    FileRequest,
    FileResponse,
    IndexerRequest,
    IndexerResponse,
)
from src.rag.indexer import index_all_from_env

app = FastAPI(title="FastAPI App", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ai/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for the AI layer.

    Flow handling:
    - If payload.form is None: treat as discovery phase (find correct form).
    - If payload.form is provided: treat as form-filling or Q&A phase.

    NOTE: AI logic intentionally omitted. This is a routing stub only.
    """

    if payload.form is None:
        # Form is unknown: guide the user to identify the correct form.
        return ChatResponse(
            message="We have not found the suitable form yet. Please provide more details.",
            form=None,
            next_field=None,
            fields=None,
        )

    # Form is known: ask for next field or answer questions.
    return ChatResponse(
        message="Please provide the next required field.",
        form=payload.form,
        next_field="next_required_field",
        fields=None,
    )


@app.post("/ai/file", response_model=FileResponse)
def file_endpoint(payload: FileRequest) -> FileResponse:
    """
    File generation endpoint.

    Called after the user confirms form completion. This should trigger PDF creation
    and storage. AI logic is intentionally omitted.
    """

    return FileResponse(
        status="pending",
        file_url=None,
        message="File generation is not implemented yet.",
    )


@app.post("/lahwita/ai/indexr/", response_model=IndexerResponse)
def indexer_endpoint(payload: IndexerRequest) -> IndexerResponse:
    """
    Index all configured Drive folders into Pinecone.
    """

    results = index_all_from_env(index_keys=payload.index_keys)
    return IndexerResponse(status="success", results=results)
