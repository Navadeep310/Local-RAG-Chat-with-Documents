"""Streamlit entry point for the local DocuMind workshop application."""

import asyncio
from pathlib import Path

import streamlit as st

from app.config import settings
from app.rag import OllamaClient, OllamaError, RAGService


st.set_page_config(page_title="DocuMind · Local RAG Workshop", page_icon="✦", layout="wide")


def run(coroutine):
    return asyncio.run(coroutine)


@st.cache_resource
def get_service() -> RAGService:
    return RAGService(OllamaClient())


def format_bytes(value: int) -> str:
    return f"{value / 1024 / 1024:.1f} MB" if value >= 1024 * 1024 else f"{max(1, value // 1024)} KB"


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f8faf9; }
        [data-testid="stSidebar"] { background: #f2f7f5; }
        h1, h2, h3 { letter-spacing: -0.04em; }
        .hero { padding: 1.5rem 0 0.5rem; }
        .eyebrow { color: #147d73; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .privacy { background:#e5f3ef; border-radius: 999px; color:#0b5f58; padding:.45rem .8rem; font-size:.78rem; display:inline-block; }
        .source { background:#e5f3ef; border-radius:.4rem; color:#0b5f58; padding:.7rem .8rem; margin:.35rem 0; font-size:.82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    render_styles()
    service = get_service()
    st.markdown('<div class="eyebrow">LOCAL RAG WORKSHOP</div>', unsafe_allow_html=True)
    st.title("Chat with your documents.")
    st.caption("Extract · Chunk · Retrieve · Generate — locally with Ollama")
    st.markdown('<span class="privacy">⌾ Your files stay on this machine</span>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Workspace")
        health = run(service.client.health())
        if health["available"] and health["embedding_model_ready"] and health["chat_model_ready"]:
            st.success("Ollama connected")
        else:
            st.warning("Ollama model setup needed")
            st.caption(f"Embedding: `{settings.ollama_embed_model}`")
            st.caption(f"Chat: `{settings.ollama_chat_model}`")
        st.divider()
        st.subheader("How it works")
        st.markdown("1. **Extract** PDF/DOCX text\n2. **Chunk** passages\n3. **Retrieve** with cosine similarity\n4. **Generate** a grounded answer")
        top_k = st.slider("Retrieved sources", min_value=1, max_value=8, value=settings.top_k)

    st.subheader("Document library")
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        processed = st.session_state.setdefault("processed_files", set())
        for uploaded_file in uploaded_files:
            file_key = f"{uploaded_file.name}:{uploaded_file.size}"
            if file_key in processed:
                continue
            if uploaded_file.size > settings.max_upload_mb * 1024 * 1024:
                st.error(f"{uploaded_file.name} is larger than {settings.max_upload_mb} MB.")
                continue
            try:
                with st.spinner(f"Embedding {uploaded_file.name}…"):
                    run(service.add_document(uploaded_file.name, uploaded_file.getvalue()))
                processed.add(file_key)
                st.success(f"Added {uploaded_file.name}")
            except (ValueError, OllamaError) as error:
                st.error(str(error))

    if service.documents:
        columns = st.columns(min(3, len(service.documents)))
        for column, document in zip(columns * ((len(service.documents) + 2) // 3), service.documents.values()):
            with column:
                st.metric(document.name, f"{len(document.chunks)} chunks", format_bytes(document.size_bytes))
    else:
        st.info("Upload a PDF or DOCX file to create your local knowledge base.")

    st.divider()
    st.subheader("Ask your documents")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View retrieved sources"):
                    for index, source in enumerate(message["sources"], start=1):
                        st.markdown(
                            f'<div class="source"><strong>Source {index}: {source["document_name"]}</strong> · '
                            f'{source["location"]} · cosine {source["score"]}<br>{source["text"]}</div>',
                            unsafe_allow_html=True,
                        )

    question = st.chat_input("Ask a question about your uploaded documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving sources and generating an answer…"):
                    answer, sources = run(service.answer(question, top_k))
                st.markdown(answer)
                with st.expander("View retrieved sources"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(
                            f'<div class="source"><strong>Source {index}: {source["document_name"]}</strong> · '
                            f'{source["location"]} · cosine {source["score"]}<br>{source["text"]}</div>',
                            unsafe_allow_html=True,
                        )
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except (ValueError, OllamaError) as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
