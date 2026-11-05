"""Tests for retrieval-augmented query behavior."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models import QueryResponse


def test_query_grounds_generation_and_returns_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify retrieved context reaches the LLM and sources are returned."""
    captured: dict[str, object] = {}

    class FakeEmbeddings:
        """Mock OpenAI's embeddings resource."""

        def create(self, model: str, input: list[str]) -> SimpleNamespace:
            """Return a deterministic query embedding."""
            assert model == "text-embedding-3-small"
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2])])

    class FakeCompletions:
        """Mock OpenAI's chat completion resource."""

        def create(self, **kwargs: object) -> SimpleNamespace:
            """Capture the prompt and return a fixed answer."""
            captured["prompt"] = kwargs["messages"]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="A grounded answer."))]
            )

    class FakeCollection:
        """Return deterministic ChromaDB results."""

        def query(self, **kwargs: object) -> dict[str, object]:
            """Capture retrieval options and return one matching chunk."""
            captured["retrieval"] = kwargs
            return {
                "documents": [["The project launched in 2024."]],
                "metadatas": [[
                    {
                        "source": "document-1",
                        "chunk_index": 0,
                        "timestamp": "2026-01-01T00:00:00+00:00",
                    }
                ]],
                "distances": [[0.12]],
            }

    class FakeOpenAI:
        """Mock the embedding and chat APIs."""

        embeddings = FakeEmbeddings()
        chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("app.main.get_openai_client", lambda: FakeOpenAI())
    monkeypatch.setattr("app.main.get_collection", lambda: FakeCollection())
    response = TestClient(app).post(
        "/query?top_k=3",
        json={"question": "When did the project launch?"},
    )

    assert response.status_code == 200
    result = QueryResponse.model_validate(response.json())
    assert result.answer == "A grounded answer."
    assert result.sources[0].text == "The project launched in 2024."
    prompt = str(captured["prompt"])
    assert "The project launched in 2024." in prompt
    assert captured["retrieval"]["n_results"] == 3


def test_empty_query_returns_structured_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify whitespace-only questions receive a structured error."""
    monkeypatch.setenv("OPENAI_API_KEY", "not-used")
    response = TestClient(app).post("/query", json={"question": "  "})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_missing_api_key_returns_service_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify requests fail clearly when OpenAI configuration is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = TestClient(app).post("/query", json={"question": "Any question?"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "missing_api_key"
