from app.adapters.opensearch import HybridRetriever, OpenSearchChunkStore
from app.domain import ACL, Chunk, RetrievalResult


class FakeIndices:
    def __init__(self) -> None:
        self.exists_result = True

    def exists(self, **kwargs):
        return self.exists_result

    def create(self, **kwargs) -> None:
        return None


class FakeOpenSearchClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {
            "hits": {
                "hits": [
                    {
                        "_score": 5.2,
                        "_source": {
                            "chunk_id": "chunk-vpn",
                            "document_id": "doc-vpn",
                            "title": "VPN 制度",
                            "text": "VPN 账号申请需要主管审批。",
                            "tenant_id": "t1",
                            "space_id": "it",
                            "allowed_subjects": ["user:bob"],
                            "source_uri": "manual://vpn",
                            "page": -1,
                            "ordinal": 0,
                            "version": 1,
                            "metadata": {},
                        },
                    }
                ]
            }
        }

    def close(self) -> None:
        return None


class StaticRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    async def retrieve(self, *args, **kwargs) -> list[RetrievalResult]:
        return self.results


def result(chunk_id: str, score: float, source: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            text=f"{chunk_id} text",
            acl=ACL(tenant_id="t1", space_id="it", allowed_subjects={"user:bob"}),
            source_uri=f"manual://{chunk_id}",
            title=chunk_id,
        ),
        score=score,
        source=source,
    )


async def test_opensearch_retriever_applies_acl_filters():
    client = FakeOpenSearchClient()
    store = OpenSearchChunkStore("http://opensearch.test:9200", "rag_chunks", client=client)

    results = await store.search_chunks(
        query="VPN 怎么申请？",
        tenant_id="t1",
        space_id="it",
        user_subjects={"user:bob", "role:employee"},
        top_k=8,
    )

    search_body = client.search_calls[0]["body"]
    filters = search_body["query"]["bool"]["filter"]
    assert {"term": {"tenant_id": "t1"}} in filters
    assert {"term": {"space_id": "it"}} in filters
    assert {"terms": {"allowed_subjects": ["role:employee", "user:bob"]}} in filters
    assert results[0].source == "opensearch-bm25"
    assert results[0].chunk.id == "chunk-vpn"


async def test_hybrid_retriever_deduplicates_with_rrf():
    vector = [result("vpn", 0.6, "milvus-vector"), result("leave", 0.5, "milvus-vector")]
    keyword = [result("vpn", 8.1, "opensearch-bm25"), result("faq", 3.2, "opensearch-bm25")]
    retriever = HybridRetriever(StaticRetriever(vector), StaticRetriever(keyword), rrf_k=60)

    results = await retriever.retrieve("VPN 怎么申请？", "t1", "it", {"user:bob"}, 3)

    assert [item.chunk.id for item in results] == ["vpn", "leave", "faq"]
    assert results[0].source == "hybrid-rrf"
    assert results[0].vector_score == 0.6
    assert results[0].score > results[1].score
