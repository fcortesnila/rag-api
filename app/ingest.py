"""Document chunking and embedding operations."""

from datetime import datetime, timezone
import re
from uuid import uuid4

import tiktoken
from chromadb.api.models.Collection import Collection
from openai import OpenAI

from app.errors import VectorStoreError

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_TOKENS = 200
OVERLAP_TOKENS = 20
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    """Return the model-compatible token count for text."""
    return len(_ENCODING.encode(text))


def _split_long_sentence(sentence: str, limit: int, overlap: int) -> list[str]:
    """Split an oversized sentence into token windows."""
    tokens = _ENCODING.encode(sentence)
    stride = max(1, limit - overlap)
    return [
        _ENCODING.decode(tokens[start : start + limit])
        for start in range(0, len(tokens), stride)
    ]


def chunk_text(
    text: str,
    max_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[str]:
    """Split text near sentence boundaries with a token overlap."""
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Chunk size must be positive and overlap smaller than the chunk size.")
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", text.strip())
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if _token_count(sentence) > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_sentence(sentence, max_tokens, overlap_tokens))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and _token_count(candidate) > max_tokens:
            chunks.append(current)
            carry = _ENCODING.encode(current)[-overlap_tokens:] if overlap_tokens else []
            room = max_tokens - _token_count(sentence)
            current = f"{_ENCODING.decode(carry[-room:])} {sentence}".strip() if room > 0 else sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def ingest_documents(
    documents: list[str],
    collection: Collection,
    client: OpenAI,
) -> int:
    """Chunk, embed, and persist documents; return the number of chunks."""
    chunks: list[str] = []
    metadatas: list[dict[str, str | int]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for document_index, document in enumerate(documents):
        for chunk_index, chunk in enumerate(chunk_text(document)):
            chunks.append(chunk)
            metadatas.append(
                {
                    "source": f"document-{document_index + 1}",
                    "chunk_index": chunk_index,
                    "timestamp": timestamp,
                }
            )
    if not chunks:
        return 0
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    embeddings = [item.embedding for item in response.data]
    try:
        collection.add(
            ids=[str(uuid4()) for _ in chunks],
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
    except Exception as exc:
        raise VectorStoreError("Could not save document chunks.") from exc
    return len(chunks)
