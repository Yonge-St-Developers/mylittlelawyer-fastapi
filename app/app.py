from fastapi import FastAPI

from app.schemas import (
    ChatRequest,
    ChatResponse,
    FileRequest,
    FileResponse,
    IndexerRequest,
    IndexerResponse,
    ConsultantRequest,
    ConsultantResponse,
)
from src.graph.builder import build_chat_graph
from src.graph.helpers import format_retrieval_context
from src.graph.file_builder import build_file_graph
from src.rag.indexer import index_all_from_env
from src.consultant.graph import build_consultant_graph

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

    retrieval = result.get("retrieval_results") or {}
    debug_request = {
        "message": payload.new_message,
        "chat_history": state.get("chat_history"),
        "form": payload.form,
        "intent": result.get("intent"),
    }
    debug_retrieval = {
        "summary": format_retrieval_context(retrieval),
        "raw_keys": list(retrieval.keys()),
    }

    return ChatResponse(
        message=message,
        form=result.get("form"),
        next_field=None,
        fields=None,
        debug_request=debug_request,
        debug_retrieval=debug_retrieval,
    )


@app.post("/ai/file", response_model=FileResponse)
def file_endpoint(payload: FileRequest) -> FileResponse:
    """
    File generation endpoint.

    Called after the user confirms form completion. This should trigger PDF creation
    and storage. AI logic is intentionally omitted.
    """

    compiled = build_file_graph()
    state = {
        "session_id": payload.session_id,
        "form_title": payload.form_title,
        "chat_history": [
            {"role": item.role, "content": item.content} for item in payload.chat_history
        ],
    }
    result = compiled.invoke(state)

    file_json = result.get("file_json") or {}
    file_name = None
    if isinstance(file_json, dict):
        file_name = file_json.get("file_name")

    return FileResponse(
        status="success",
        file_name=file_name,
        file_json=file_json,
        message=None,
    )


@app.post("/lahwita/ai/indexr/", response_model=IndexerResponse)
def indexer_endpoint(payload: IndexerRequest) -> IndexerResponse:
    """
    Index all configured Drive folders into Pinecone.
    """

    results = index_all_from_env(index_keys=payload.index_keys)
    return IndexerResponse(status="success", results=results)


@app.post("/ai/consultant", response_model=ConsultantResponse)
def consultant_endpoint(payload: ConsultantRequest) -> ConsultantResponse:
    """
    Consultant endpoint for agentic RAG (CanLII cases).
    """

    compiled = build_consultant_graph()
    state = {
        "message": payload.message,
        "chat_history": [
            {"role": item.role, "content": item.content}
            for item in (payload.chat_history or [])
        ],
        "refresh_index": payload.refresh_index or False,
    }
    result = compiled.invoke(state)

    return ConsultantResponse(answer=result.get("response_text", ""))
