"""Prompt templates for the RAG conversation flow."""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

# 1) Determine the user's current need and whether a form is required
FORM_INTENT_PROMPT = PromptTemplate(
    input_variables=["message"],
    template=(
        "You are a legal intake assistant for Ontario LTB matters. "
        "Classify the user's message into ONE primary intent:\n"
        "1) answering_form_field\n"
        "2) asking_about_form\n"
        "3) describing_situation\n"
        "4) other\n\n"
        "Then decide whether a specific LTB application form is likely needed.\n\n"
        "User message: {message}\n\n"
        "Return a concise JSON-like object with keys:\n"
        "intent (one of the 4 above),\n"
        "need_form (yes/no),\n"
        "form_hint (short guess or null),\n"
        "why (1 short sentence)."
    ),
)

# 2) Ask clarifying questions to identify the correct form
FORM_DISCOVERY_PROMPT = PromptTemplate(
    input_variables=["message", "current_hint"],
    template=(
        "You are helping identify the correct LTB application. "
        "Be polite, clear, and precise. Ask only ONE concise clarifying question. "
        "Do not provide multiple options unless necessary.\n\n"
        "User message: {message}\n"
        "Current form hint: {current_hint}\n\n"
        "Ask the single best question to determine the correct form."
    ),
)

# 3) Answer user questions about the form
FORM_QA_PROMPT = PromptTemplate(
    input_variables=["message", "form_title", "context"],
    template=(
        "You are answering questions about the LTB form: {form_title}. "
        "Be polite, clear, and exact. Only use the provided context. "
        "If the answer is not in the context, say you don't have enough information.\n\n"
        "Relevant context:\n{context}\n\n"
        "User question: {message}\n\n"
        "Provide a concise, accurate answer."
    ),
)

# 4) Ask for the next required field during form filling
FORM_FILL_PROMPT = PromptTemplate(
    input_variables=["form_title", "known_fields", "remaining_fields"],
    template=(
        "You are guiding the user to fill the LTB form: {form_title}. "
        "Be polite, clear, and exact. Ask for only ONE field at a time. "
        "If the field has a required format (date, amount, address), state the format.\n\n"
        "Known fields: {known_fields}\n"
        "Remaining fields: {remaining_fields}\n\n"
        "Ask for the next single most important field."
    ),
)

# 5) Final confirmation before PDF generation
FORM_CONFIRM_PROMPT = PromptTemplate(
    input_variables=["form_title", "filled_fields"],
    template=(
        "You are about to finalize the LTB form: {form_title}.\n\n"
        "Filled fields summary:\n{filled_fields}\n\n"
        "Ask the user for final confirmation to generate the PDF."
    ),
)

# 6) Build a JSON payload for PDF generation
FORM_FILE_JSON_PROMPT = PromptTemplate(
    input_variables=["form_title", "chat_history"],
    template=(
        "You are preparing a JSON payload to generate the LTB form: {form_title}. "
        "Use only the chat history to infer field values. "
        "If a value is missing or unclear, leave it null and list the field name in missing_fields. "
        "Return ONLY valid JSON with this exact shape:\n"
        "{\n"
        '  "file_name": "<suggested filename ending with .pdf>",\n'
        '  "form_title": "<form title>",\n'
        '  "fields": {\n'
        '    "<field_key>": "<value or null>"\n'
        "  },\n"
        '  "missing_fields": ["<field_key>", "..."],\n'
        '  "assumptions": ["<short assumption>", "..."]\n'
        "}\n\n"
        "Chat history:\n{chat_history}\n\n"
        "Return only JSON. No extra text."
    ),
)
