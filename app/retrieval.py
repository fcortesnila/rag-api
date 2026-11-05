"""ChromaDB retrieval operations."""

from typing import Any

from chromadb.api.models.Collection import Collection
from openai import OpenAI

from app.errors import VectorStoreError
from app.ingest import EMBEDDING_MODEL
from app.models import SourceChunk


def retrieve_chunks(
    question: str,
    top_k: int,
    collection: Collection,
    client: OpenAI,
) -> list[SourceChunk]:
    """Embed a question and return its nearest stored chunks."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    try:
        result: dict[str, Any] = collection.query(
            query_embeddings=[response.data[0].embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise VectorStoreError("Could not retrieve document chunks.") from exc
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        SourceChunk(
            source=str(metadata["source"]),
            chunk_index=int(metadata["chunk_index"]),
            timestamp=str(metadata["timestamp"]),
            text=str(document),
            distance=float(distances[index]) if index < len(distances) else None,
        )
        for index, (document, metadata) in enumerate(zip(documents, metadatas))
    ]
