"""Prompt chaining utilities for conditional prompt selection."""

from __future__ import annotations

from typing import Literal

from src.prompts.templates import (
    FORM_CONFIRM_PROMPT,
    FORM_DISCOVERY_PROMPT,
    FORM_FILL_PROMPT,
    FORM_INTENT_PROMPT,
    FORM_QA_PROMPT,
)


IntentType = Literal[
    "answering_form_field",
    "asking_about_form",
    "describing_situation",
    "other",
]


def select_next_prompt(intent: IntentType, form_known: bool) -> str:
    """
    Select the next prompt template name based on intent and form status.

    This is a simple router for the conditional graph:
    - If form is unknown, discovery has priority.
    - If user is asking about the form, use QA prompt.
    - If user is answering a form field, use fill prompt.
    - Otherwise, fall back to discovery.
    """

    if not form_known:
        return "FORM_DISCOVERY_PROMPT"

    if intent == "asking_about_form":
        return "FORM_QA_PROMPT"

    if intent == "answering_form_field":
        return "FORM_FILL_PROMPT"

    return "FORM_DISCOVERY_PROMPT"


def get_prompt_template(name: str):
    """Return the prompt template by name."""

    templates = {
        "FORM_INTENT_PROMPT": FORM_INTENT_PROMPT,
        "FORM_DISCOVERY_PROMPT": FORM_DISCOVERY_PROMPT,
        "FORM_QA_PROMPT": FORM_QA_PROMPT,
        "FORM_FILL_PROMPT": FORM_FILL_PROMPT,
        "FORM_CONFIRM_PROMPT": FORM_CONFIRM_PROMPT,
    }
    if name not in templates:
        raise ValueError(f"Unknown prompt template: {name}")
    return templates[name]
