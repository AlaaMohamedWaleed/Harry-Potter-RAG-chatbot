# Harry Potter RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural-language questions about the 7 Harry Potter books, with source citations (book name + page number) and a simple web UI.

## How It Works

1. **Data Prep (Kaggle notebook)** — the 7-book PDF is parsed into per-page Markdown files, cleaned, tagged with book name + page number, embedded using `intfloat/multilingual-e5-small`, and upserted into a Qdrant Cloud vector collection.
2. **FastAPI backend (`rag_api.py`)** — on each user query:
   - A Groq LLM call **routes** the query into `retrieve`, `chitchat`, or `off-topic`.
   - `retrieve` → the query is embedded, the top-k most similar pages are fetched from Qdrant, and a Gemini LLM generates an answer grounded only in that retrieved context.
   - `chitchat` → a lightweight, friendly reply with no book lookup.
   - `off-topic` → a fixed refusal message.
3. **Frontend (`index.html`)** — a static page that calls the FastAPI server directly: a health-check button, a question box, and a display for the answer + its sources.

## Architecture

```
PDF (7 books)
   │  (Kaggle: parse → clean → chunk by page → embed)
   ▼
Qdrant Cloud (vector store)
   ▲
   │  (query embedding + similarity search)
FastAPI (rag_api.py) ── router (Groq) ── generator (Gemini)
   ▲
   │  fetch()
index.html (UI)
```

## Repo Contents

| File | Purpose |
|---|---|
| `creating-and-uploading-embeddings.ipynb` | Kaggle notebook: PDF → Markdown → cleaning → chunking → embeddings → Qdrant upload |
| `rag_api.py` | FastAPI server implementing the routing + RAG pipeline |
| `index.html` | Frontend UI (health check, ask a question, view sources) |
| `requirements.txt` | Python dependencies for running `rag_api.py` locally |
| `.env.example` | Template for required environment variables (copy to `.env` and fill in) |
| `screenshots/` | UI screenshots demonstrating the working app |

## Setup

### 1. Vector store (one-time)
Run `creating-and-uploading-embeddings.ipynb` in a Kaggle notebook (GPU not required — embeddings use `e5-small` on CPU or GPU) with your Qdrant Cloud credentials set as Kaggle Secrets (`QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`). This populates your Qdrant Cloud collection with the embedded book pages.

### 2. Backend
```bash
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`) with:
```
QDRANT_URL=your-qdrant-cloud-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=your-collection-name
EMBEDDING_MODEL=intfloat/multilingual-e5-small
GEMINI_MODEL=gemini-3.6-flash
GEMINI_API_KEY=your-gemini-api-key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your-groq-api-key
TOP_K=5
```

Run the server:
```bash
uvicorn rag_api:app --port 8001
```

API docs available at `http://127.0.0.1:8001/docs`.

### 3. Frontend
Open `index.html` directly in a browser. Set the API URL field to `http://127.0.0.1:8001`, click **Ping API** to confirm the connection, then ask a question.

## Example Queries

- **Retrieve route**: "What is the Sorting Hat and how does it decide which house a student belongs to?"
- **Chitchat route**: "Hi, do you like Harry Potter?"
- **Off-topic route**: "What's the capital of France?"

## Screenshots

See the `screenshots/` folder for the UI in action: health check, a grounded answer with sources, chitchat, and off-topic handling.

## Known Limitations

- Chunking is done at the **whole-page** level rather than smaller semantic chunks. This works well for facts described across multiple sentences/paragraphs, but can occasionally miss short, single-line facts if they're embedded on a page whose overall content is about something else (lower vector similarity to the query). A smaller, overlapping chunking strategy would likely improve recall on such queries.
- The router and chitchat responses depend on the LLM's classification accuracy; edge-case phrasing may occasionally be misrouted.
