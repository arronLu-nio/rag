"""DashScope qwen3-vl-rerank adapter."""

from typing import Any

import httpx

from app.domain import RetrievalResult
from app.ports.contracts import Reranker


class DashScopeReranker(Reranker):
    """调用百炼 qwen3-vl-rerank，对 Milvus 候选 Chunk 做二次相关性判断。"""

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        payload = {
            "model": self.model,
            "input": {
                "query": {"text": query},
                "documents": [{"text": result.chunk.text} for result in results],
            },
            "parameters": {
                "return_documents": False,
                "top_n": len(results),
                "instruct": (
                    "Given a user question, rank passages by whether they contain evidence "
                    "that directly answers the question. Assign low relevance to merely topical "
                    "or keyword-overlapping passages."
                ),
            },
        }
        response_body = await self._post(payload)
        if response_body.get("code"):
            raise RuntimeError(
                f"DashScope rerank failed: {response_body['code']}: "
                f"{response_body.get('message', 'unknown error')}"
            )

        reranked: list[RetrievalResult] = []
        for item in response_body.get("output", {}).get("results", []):
            index = int(item["index"])
            if index < 0 or index >= len(results):
                raise RuntimeError(f"DashScope rerank returned invalid document index: {index}")
            result = results[index]
            result.rerank_score = float(item["relevance_score"])
            reranked.append(result)

        reranked.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        print(f"============={self.model} rerank 分数=============")
        for rank, result in enumerate(reranked, start=1):
            print(
                f"rank={rank} retrieval_score={result.score:.4f} "
                f"rerank_score={result.rerank_score:.4f} "
                f"chunk_id={result.chunk.id} source_uri={result.chunk.source_uri}"
            )
        print("========================================")
        return reranked

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.client is not None:
            response = await self.client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
