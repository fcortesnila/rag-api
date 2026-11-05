"""FastAPI routes for document ingestion and RAG queries."""

import logging
import os

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi import Request
from fastapi.responses import JSONResponse
from openai import OpenAI

from app.errors import VectorStoreError
from app.generation import generate_answer
from app.ingest import ingest_documents
from app.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.retrieval import retrieve_chunks

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="RAG API", version="1.0.0")


def get_openai_client() -> OpenAI:
    """Create an OpenAI client or report missing configuration."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "missing_api_key",
                "message": "Set OPENAI_API_KEY to use ingestion and query endpoints.",
            },
        )
    return OpenAI(api_key=api_key)


def get_collection() -> chromadb.Collection:
    """Open the persistent application collection."""
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        return client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:
        raise VectorStoreError("Could not open the ChromaDB collection.") from exc


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return request validation errors in the API's structured format."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request body is invalid.",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Index the submitted documents and report chunk totals."""
    openai_client = get_openai_client()
    try:
        collection = get_collection()
        count = ingest_documents(request.documents, collection, openai_client)
    except VectorStoreError as exc:
        logger.exception("Vector store ingestion failed")
        raise HTTPException(
            status_code=503,
            detail={"code": "vector_store_error", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("OpenAI ingestion request failed")
        raise HTTPException(
            status_code=502,
            detail={"code": "upstream_error", "message": "Document embedding failed."},
        ) from exc
    return IngestResponse(indexed_documents=len(request.documents), indexed_chunks=count)


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    top_k: int = Query(default=3, ge=1),
) -> QueryResponse:
    """Retrieve relevant context and generate a grounded answer."""
    openai_client = get_openai_client()
    try:
        collection = get_collection()
        chunks = retrieve_chunks(request.question, top_k, collection, openai_client)
    except VectorStoreError as exc:
        logger.exception("Vector store query failed")
        raise HTTPException(
            status_code=503,
            detail={"code": "vector_store_error", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("OpenAI query embedding failed")
        raise HTTPException(
            status_code=502,
            detail={"code": "upstream_error", "message": "Question embedding failed."},
        ) from exc
    try:
        answer = generate_answer(request.question, chunks, openai_client)
    except Exception as exc:
        logger.exception("OpenAI answer generation failed")
        raise HTTPException(
            status_code=502,
            detail={"code": "upstream_error", "message": "Answer generation failed."},
        ) from exc
    return QueryResponse(answer=answer, sources=chunks)
