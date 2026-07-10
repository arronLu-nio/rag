from types import SimpleNamespace

import pytest

from app.adapters.dashscope import DashScopeEmbeddingModel
from app.main import build_embedding_model


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2, 0.3]),
                SimpleNamespace(embedding=[0.4, 0.5, 0.6]),
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


async def test_dashscope_embedding_uses_openai_compatible_api():
    client = FakeOpenAIClient()
    model = DashScopeEmbeddingModel(
        api_key="test-key",
        api_base="https://example.test/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=1024,
        client=client,
    )

    embeddings = await model.embed(["第一段文本", "第二段文本"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert client.embeddings.calls == [
        {
            "model": "text-embedding-v4",
            "input": ["第一段文本", "第二段文本"],
            "dimensions": 1024,
            "encoding_format": "float",
        }
    ]


async def test_dashscope_embedding_returns_empty_list_for_empty_input():
    client = FakeOpenAIClient()
    model = DashScopeEmbeddingModel(
        api_key="test-key",
        api_base="https://example.test/compatible-mode/v1",
        model="text-embedding-v4",
        dimensions=1024,
        client=client,
    )

    assert await model.embed([]) == []
    assert client.embeddings.calls == []


def test_empty_embedding_key_prevents_invalid_startup(monkeypatch):
    monkeypatch.setattr("app.main.settings.embedding_provider", "dashscope")
    monkeypatch.setattr("app.main.settings.embedding_api_key", "")

    with pytest.raises(RuntimeError, match="DashScope embedding"):
        build_embedding_model()
