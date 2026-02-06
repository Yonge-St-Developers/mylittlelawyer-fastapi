from fastapi import FastAPI

from app.schemas import (
    ChatRequest,
    ChatResponse,
    FileRequest,
    FileResponse,
    IndexerRequest,
    IndexerResponse,
)
from src.graph.builder import build_chat_graph
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

    compiled = build_chat_graph()
    state = {
        "session_id": payload.session_id,
        "new_message": payload.new_message,
        "chat_history": [
            {"role": item.role, "content": item.content}
            for item in (payload.chat_history or [])
        ],
        "form": payload.form,
    }

    result = compiled.invoke(state)
    message = result.get("response_text") or ""

    return ChatResponse(
        message=message,
        form=result.get("form"),
        next_field=None,
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
