from typing import Any

from openai import AsyncOpenAI

from app.ports.contracts import EmbeddingModel


class DashScopeEmbeddingModel(EmbeddingModel):
    """百炼 embedding 适配器。

    百炼提供 OpenAI-compatible API，所以这里直接使用 OpenAI SDK。
    这个类只负责把文本列表转成向量列表，不参与切分、检索或权限判断。
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        dimensions: int,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.client = client or AsyncOpenAI(api_key=api_key, base_url=api_base)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
            encoding_format="float",
        )
        embeddings = [item.embedding for item in response.data]
        if embeddings:
            preview = embeddings[0][:8]
            print(
                f"=============embedding {self.model} "
                f"dimensions={len(embeddings[0])} first_8={preview}============="
            )
        return embeddings
