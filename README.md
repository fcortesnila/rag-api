# FastAPI RAG API

This project is a small, self-contained Retrieval-Augmented Generation API: it indexes text in a local ChromaDB collection and answers questions with relevant retrieved passages, making each response inspectable and grounded in source chunks.

## Run locally

These steps are only for running the app directly on your computer. If you use Docker, skip this section; you do not need to create a local virtual environment or install Python dependencies on your computer.

1. Create and activate a virtual environment, then install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your OpenAI API key:

   ```bash
   cp .env.example .env
   ```

3. Start the development server:

   ```bash
   uvicorn app.main:app --reload
   ```

## Run with Docker

1. Copy `.env.example` to `.env` and add your OpenAI API key.
2. Build and start the service:

   ```bash
   docker compose up --build
   ```

Once the app is running, open [http://localhost:8000/docs](http://localhost:8000/docs) to use the API. The app can start without an API key, but `/ingest` and `/query` require `OPENAI_API_KEY` and return `503` if it is missing.

The Compose setup keeps ChromaDB data in the `chroma_data` named volume. Stop the service with `docker compose down`; the data remains available on the next start. To remove the stored data as well, run `docker compose down --volumes`.

## Local files and Git

The `.gitignore` rules keep local secrets (`.env`), Python virtual environments and caches, test/tool caches, and the local `chroma_db/` data directory out of Git. The example configuration `.env.example` is intentionally tracked; never put a real API key in it.

## Example requests

Index two documents:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"documents":["FastAPI is a Python web framework.","ChromaDB stores vector embeddings locally."]}'
```

Ask a question (the `top_k` query parameter defaults to 3):

```bash
curl -X POST "http://localhost:8000/query?top_k=3" \
  -H "Content-Type: application/json" \
  -d '{"question":"What does ChromaDB store?"}'
```

The query response includes the generated `answer` and a `sources` array containing the text and metadata for each retrieved chunk.

## How the RAG pattern works

**Embed → retrieve → ground → generate:** documents and questions are converted to embeddings; ChromaDB retrieves nearby document chunks by cosine distance; those chunks are supplied as context to the language model; the model generates an answer using that context.

## How I built this

OpenAI Codex/Claude Code was used as the agentic development tool to implement the readable, direct FastAPI, OpenAI, and ChromaDB pipeline without a RAG framework. I manually verified the route contracts, local persistence configuration, required error behavior, and GitHub Actions workflow. The tests mock the OpenAI embeddings and chat APIs and ChromaDB operations; they cover chunk indexing metadata, retrieval options, context inclusion in the generation prompt, response-model validation, and empty-query validation without making network calls.
