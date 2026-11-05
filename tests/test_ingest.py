"""Tests for document chunking and ingestion."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.main import app


def test_ingest_embeds_chunks_and_saves_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify embedding output and chunk metadata reach ChromaDB."""
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        """Mock OpenAI's embeddings resource."""

        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            """Return one deterministic embedding per input chunk."""
            assert model == "text-embedding-3-small"
            captured["embedded"] = input
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[float(index)]) for index in range(len(input))]
            )

    class FakeCollection:
        """Capture the ChromaDB add call."""

        def add(self, **kwargs: object) -> None:
            """Record indexed values for assertions."""
            captured.update(kwargs)

    class FakeOpenAI:
        """Mock OpenAI client for ingestion."""

        embeddings = FakeEmbeddings()

    monkeypatch.setattr("app.main.get_openai_client", lambda: FakeOpenAI())
    monkeypatch.setattr("app.main.get_collection", lambda: FakeCollection())
    response = TestClient(app).post(
        "/ingest",
        json={"documents": ["First sentence. Second sentence.", "Third sentence."]},
    )

    assert response.status_code == 200
    assert response.json() == {"indexed_documents": 2, "indexed_chunks": 2}
    assert len(captured["ids"]) == 2
    assert len(captured["embeddings"]) == 2
    metadata = captured["metadatas"]
    assert [item["source"] for item in metadata] == ["document-1", "document-2"]
    assert [item["chunk_index"] for item in metadata] == [0, 0]
    assert all(item["timestamp"] for item in metadata)
