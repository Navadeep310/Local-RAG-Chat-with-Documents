# DocuMind — Local RAG Workshop

DocuMind is a small, inspectable “Chat with Documents” application for teaching the core Retrieval-Augmented Generation loop:

1. **Extract** meaningful text from PDF or DOCX files.
2. **Chunk** text into overlapping word windows.
3. **Embed** each chunk with Ollama (`nomic-embed-text`).
4. **Retrieve** the most relevant chunks using cosine similarity.
5. **Generate** a grounded answer with a local Ollama chat model.

The application stores documents and embeddings in memory, so restarting the server clears the workshop workspace. No uploaded content leaves your machine.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running
- Models pulled locally:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate

python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The status indicator in the top-right confirms whether both configured Ollama models are available.

Configuration can be changed with environment variables; see `.env.example`. For example, to use a different chat model:

```powershell
$env:OLLAMA_CHAT_MODEL="qwen2.5:3b"
python -m uvicorn app.main:app --reload
```

## Workshop tour

- `app/text_processing.py` demonstrates extraction and overlap-aware chunking.
- `app/rag.py` contains the Ollama client, cosine similarity implementation, retrieval, and grounded prompt.
- `app/main.py` exposes a small FastAPI API and serves the frontend.
- `app/static/` is a framework-free frontend that makes every step visible to beginners.
- `tests/test_rag.py` shows how the retrieval behavior can be tested without a running Ollama server.

## Checks

```bash
python -m pytest -q
python -m compileall app
```

The API docs are available at <http://127.0.0.1:8000/docs> while the server is running.

## Streamlit mode

If you prefer a single-file workshop interface, run:

```bash
python -m streamlit run streamlit_app.py
```

Streamlit will open <http://localhost:8501>. It uses the same extraction, embedding, retrieval, and Ollama generation code as the FastAPI interface.

The Streamlit app is intended to run locally because Ollama is local. A public Streamlit Community Cloud URL cannot reach an Ollama server running on your laptop unless you provide a securely reachable model endpoint or replace the local Ollama client.

## Technical notes and limitations

- The default chunks contain up to **220 words**, with **40 words of overlap**. Words are not tokens; token counts depend on the model. PDF chunks stay within their original page. DOCX paragraphs and table rows are extracted in document order.
- Cosine similarity is `dot(query, chunk) / (norm(query) * norm(chunk))`. It measures vector direction, not answer correctness or confidence. The app ranks all passages and takes the top K; it does not apply a universal relevance threshold.
- Expand any source underneath an answer to inspect the actual retrieved text. The prompt asks the model to cite sources and abstain when evidence is missing. Grounding reduces hallucinations but does not guarantee factual answers: compare answers with the excerpts.
- Scanned/image-only PDFs need OCR in another tool before uploading. Complex PDF columns, charts, equations, and DOCX images may not extract accurately.
- The chat display retains turns, but each question is processed independently. Ask complete questions rather than pronoun-only follow-ups.
- This is a single-user, in-memory workshop app. Use one server worker, keep it bound to localhost, and expect uploads to disappear on restart or development reload. A shared deployment would need per-user indexes and durable storage.
- `.env.example` documents environment variables; to load a copied `.env`, start with `python -m uvicorn app.main:app --env-file .env`. Otherwise set variables in your shell.

## Suggested 60-minute workshop

1. **0–10 min:** Start Ollama and the app; upload a short handout and ask a factual question.
2. **10–20 min:** Inspect `extract_text` and `chunk_segments`. Explain why overlap preserves boundary information.
3. **20–35 min:** Walk through cosine similarity. Call `/api/retrieve` in `/docs` to inspect scores without generating an answer.
4. **35–45 min:** Trace the retrieved excerpts into the prompt. Expand sources and compare the answer with the original document.
5. **45–55 min:** Ask an out-of-document question; discuss abstention, irrelevant retrieval, and hallucinations. Try top K of 3 versus 6.
6. **55–60 min:** Run the tests. Experiment with chunk overlap after restarting and re-uploading documents.

**Demo content:** Create a PDF or DOCX stating “The library opens at 8 AM on weekdays. Students may borrow three books for fourteen days.” Ask “How many books may students borrow?” Then ask “Who designed the library?” to examine unsupported-question behavior.
