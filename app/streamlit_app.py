import base64
import json
import os
from typing import Any, Dict, List, Optional

import httpx
import streamlit as st

API_URL = os.getenv("FASTAPI_URL", "http://api:8000")

st.set_page_config(page_title="Lahwita Chat", page_icon="💬")

st.title("Lahwita Chat")
st.caption("Simple Streamlit client for /ai/chat and /ai/file")

if "session_id" not in st.session_state:
    st.session_state.session_id = "session-demo"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "form" not in st.session_state:
    st.session_state.form = None


def call_chat(message: str) -> Dict[str, Any]:
    payload = {
        "session_id": st.session_state.session_id,
        "new_message": message,
        "chat_history": st.session_state.chat_history or None,
        "form": st.session_state.form,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_URL}/ai/chat", json=payload)
        resp.raise_for_status()
        return resp.json()


def call_file() -> Dict[str, Any]:
    payload = {
        "session_id": st.session_state.session_id,
        "chat_history": st.session_state.chat_history,
        "form_title": st.session_state.form or "unknown",
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{API_URL}/ai/file", json=payload)
        resp.raise_for_status()
        return resp.json()


with st.sidebar:
    st.subheader("Session")
    st.text_input("Session ID", value=st.session_state.session_id, key="session_id")
    st.text_input("Form (optional)", value=st.session_state.form or "", key="form_input")
    if st.session_state.form_input.strip() == "":
        st.session_state.form = None
    else:
        st.session_state.form = st.session_state.form_input.strip()

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.form = None


st.subheader("Chat")

for msg in st.session_state.chat_history:
    role = msg.get("role", "user")
    with st.chat_message(role):
        st.write(msg.get("content", ""))

user_message = st.chat_input("Type your message")

if user_message:
    st.session_state.chat_history.append({"role": "user", "content": user_message})

    with st.chat_message("assistant"):
        try:
            result = call_chat(user_message)
            response_text = result.get("message", "")
            st.write(response_text)

            if result.get("form"):
                st.session_state.form = result["form"]

            st.session_state.chat_history.append({"role": "assistant", "content": response_text})
        except Exception as exc:
            st.error(f"Chat request failed: {exc}")


st.divider()

st.subheader("PDF Downloader")
if st.button("Generate PDF"):
    try:
        result = call_file()
        if result.get("file_base64"):
            file_bytes = base64.b64decode(result["file_base64"])
            file_name = result.get("file_name", "form.pdf")
            st.download_button(
                label="Download PDF",
                data=file_bytes,
                file_name=file_name,
                mime="application/pdf",
            )
        else:
            st.info(result.get("message", "No file available yet."))
    except Exception as exc:
        st.error(f"File request failed: {exc}")
