

import sys
from pathlib import Path

# Make `src` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.rag.chat import answer_question
from src.rag.config import get_settings

st.set_page_config(page_title="Document Assistant", page_icon="📄")

try:
    _cfg = get_settings()
    st.sidebar.markdown("**Running on**")
    st.sidebar.caption(f"Models: {_cfg.provider} ({_cfg.chat_model})")
    st.sidebar.caption(f"Vector store: {_cfg.vector_store}")
except Exception as exc:  # config problems are shown clearly instead of a stack trace
    st.sidebar.error(str(exc))

st.title("📄 Document Assistant")
st.caption("Ask questions about your documents. Answers cite the file and page they came from.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def _show_chunks(chunks: list[dict]) -> None:
    for i, s in enumerate(chunks, start=1):
        label = s.get("ref", i)  # keep the same number the answer used, e.g. [1]
        st.markdown(f"**[{label}] {s['source']}, page {s['page']}**")
        st.caption(s["content"][:400] + ("…" if len(s["content"]) > 400 else ""))


def render_sources(sources: list[dict], retrieved: list[dict], cited: bool) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            if not cited:
                st.caption(
                    "The model didn't cite its sources, so these are the retrieved chunks "
                    "that best match its answer."
                )
            _show_chunks(sources)
    # Everything the search returned, for debugging retrieval quality.
    if retrieved and len(retrieved) > len(sources):
        with st.expander(f"All retrieved chunks ({len(retrieved)})"):
            _show_chunks(retrieved)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(
                message.get("sources", []), message.get("retrieved", []), message.get("cited", True)
            )

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = answer_question(question)
            except Exception as exc:  # show config/network problems in the UI
                st.error(f"Something went wrong: {exc}")
                st.stop()
        st.markdown(result["answer"])
        render_sources(result["sources"], result["retrieved"], result["cited"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "retrieved": result["retrieved"],
            "cited": result["cited"],
        }
    )
