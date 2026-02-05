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

    This returns a file reference or status. The AI generation step is left blank
    per instructions.
    """

    status: str = Field(..., description="Status of the request: success/pending/error")
    file_name: Optional[str] = Field(
        default=None,
        description="Generated PDF filename (e.g., form_a1.pdf).",
    )
    file_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded PDF bytes to send directly to Django.",
    )
    message: Optional[str] = Field(
        default=None,
        description="Additional info for the client.",
    )
