from typing import Any, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Single chat message item in the history."""

    role: str = Field(..., description="Role of the sender: user/assistant/system")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    """
    Chat endpoint request payload.

    This single endpoint handles both:
    1) First-message flow (form is still unknown)
    2) Ongoing chat flow (form is known and user is filling it)
    """

    session_id: str = Field(..., description="Unique session ID from Django/Frontend")
    new_message: str = Field(..., description="User's latest message")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional chat history to keep context",
    )
    form: Optional[str] = Field(
        default=None,
        description="Form identifier or title. If null, system must find the right form.",
    )


class ChatResponse(BaseModel):
    """
    Chat endpoint response payload.

    - If form is still unknown, response should guide user until form is found.
    - If form is known, response should ask for the next field or answer the user's question.
    """

    message: str = Field(..., description="Assistant response text")
    form: Optional[str] = Field(
        default=None,
        description="Resolved form identifier/title. Null if not yet determined.",
    )
    next_field: Optional[str] = Field(
        default=None,
        description="Next form field requested (if user is filling a form).",
    )
    fields: Optional[List[dict[str, Any]]] = Field(
        default=None,
        description="Optional structured field hints or metadata.",
    )
    debug_request: Optional[dict[str, Any]] = Field(
        default=None,
        description="Debug payload showing what was sent to the model.",
    )
    debug_retrieval: Optional[dict[str, Any]] = Field(
        default=None,
        description="Debug payload showing retrieved data (if available).",
    )


class FileRequest(BaseModel):
    """
    File endpoint request payload.

    This endpoint is called when the user confirms form completion and we need
    to generate/store the final document.
    """

    session_id: str = Field(..., description="Unique session ID from Django/Frontend")
    chat_history: List[ChatMessage] = Field(..., description="Full chat history")
    form_title: str = Field(..., description="Final form title or ID")


class FileResponse(BaseModel):
    """
    File endpoint response payload.

    This returns the JSON payload for PDF generation.
    """

    status: str = Field(..., description="Status of the request: success/pending/error")
    file_name: Optional[str] = Field(
        default=None,
        description="Generated PDF filename (e.g., form_a1.pdf).",
    )
    file_json: Optional[dict[str, Any]] = Field(
        default=None,
        description="JSON payload with form fields for PDF generation.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional info for the client.",
    )


class IndexerRequest(BaseModel):
    """Indexer endpoint request payload."""

    index_keys: Optional[List[str]] = Field(
        default=None,
        description="Optional list of index keys to run (e.g., rta_act, ltb_forms).",
    )


class IndexerResponse(BaseModel):
    """Indexer endpoint response payload."""

    status: str = Field(..., description="Status of the request: success/error")
    results: Optional[dict[str, Any]] = Field(
        default=None,
        description="Per-index results for the indexing run.",
    )


class ConsultantRequest(BaseModel):
    """Consultant endpoint request payload."""

    message: str = Field(..., description="User question")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional chat history for context",
    )
    refresh_index: Optional[bool] = Field(
        default=False,
        description="If true, crawl and refresh CanLII index before retrieval",
    )


class ConsultantResponse(BaseModel):
    """Consultant endpoint response payload."""

    answer: str = Field(..., description="Assistant response")
