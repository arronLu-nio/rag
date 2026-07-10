from types import SimpleNamespace

from app.adapters.dashscope_reranker import DashScopeReranker
from app.domain import ACL, Chunk, RetrievalResult


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.12},
                ]
            }
        }


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(SimpleNamespace(url=url, **kwargs))
        return FakeResponse()


def result(chunk_id: str, text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            id=chunk_id,
            document_id="doc-1",
            text=text,
            acl=ACL(tenant_id="t1", space_id="it", allowed_subjects={"user:bob"}),
            source_uri="manual://it",
            title="IT制度",
        ),
        score=0.3,
        source="milvus-vector",
    )


async def test_dashscope_reranker_maps_scores_back_to_original_chunks():
    client = FakeHttpClient()
    reranker = DashScopeReranker(
        api_key="test-key",
        api_url="https://example.test/rerank",
        model="qwen3-vl-rerank",
        client=client,
    )

    reranked = await reranker.rerank(
        "VPN 怎么申请？",
        [result("chunk-a", "考勤申请流程"), result("chunk-b", "VPN 账号申请流程")],
    )

    assert [item.chunk.id for item in reranked] == ["chunk-b", "chunk-a"]
    assert [item.rerank_score for item in reranked] == [0.93, 0.12]
    assert client.calls[0].url == "https://example.test/rerank"
    assert client.calls[0].headers["Authorization"] == "Bearer test-key"
    assert client.calls[0].json["input"]["query"] == {"text": "VPN 怎么申请？"}
    assert client.calls[0].json["input"]["documents"] == [
        {"text": "考勤申请流程"},
        {"text": "VPN 账号申请流程"},
    ]
