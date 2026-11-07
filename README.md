# FastAPI RAG API

This project is a small, self-contained Retrieval-Augmented Generation API: it indexes text in a local ChromaDB collection and answers questions with relevant retrieved passages, making each response inspectable and grounded in source chunks.

## Setup

1. Clone the repository and enter the project directory.
2. Create a `.env` file from `.env.example` and add your OpenAI API key.
3. Install dependencies with `pip install -r requirements.txt`.
4. Start the API with `uvicorn app.main:app --reload`.

The app starts even if `OPENAI_API_KEY` is absent. In that case, `/ingest` and `/query` return a clear `503` error when called.

## Run with Docker

1. Copy `.env.example` to `.env` and add your OpenAI API key (in PowerShell, run `Copy-Item .env.example .env`, then edit `.env`). The key is optional for starting the API, but required for `/ingest` and `/query`.
2. Build and start the service:

   ```bash
   docker compose up --build
   ```

3. Open [http://localhost:8000/docs](http://localhost:8000/docs) to use the API.

The Compose setup keeps ChromaDB data in the `chroma_data` named volume. Stop the service with `docker compose down`; the data remains available on the next start. To remove the stored data as well, run `docker compose down --volumes`.

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
