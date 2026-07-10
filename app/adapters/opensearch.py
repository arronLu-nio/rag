"""OpenSearch keyword indexing and BM25 retrieval adapters."""

import asyncio
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from app.domain import ACL, Chunk, RetrievalResult
from app.ports.contracts import Retriever


class OpenSearchChunkStore:
    """把 Chunk 文本、来源和 ACL 写入 OpenSearch，供 BM25 关键词检索。"""

    def __init__(self, url: str, index_name: str, client: Any | None = None) -> None:
        self.url = url
        self.index_name = index_name
        self._client = client
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        await self.initialize()
        await asyncio.to_thread(
            bulk,
            self._client,
            [self._index_action(chunk) for chunk in chunks],
            refresh=True,
        )

    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        """写入新 Chunk 后，删除当前文档的旧关键词索引。"""

        await self.initialize()
        if not chunks:
            await self.delete_chunks(document_id)
            return
        await self.save_chunks(chunks)
        new_chunk_ids = [chunk.id for chunk in chunks]
        await asyncio.to_thread(
            self._client.delete_by_query,
            index=self.index_name,
            body={
                "query": {
                    "bool": {
                        "filter": [{"term": {"document_id": document_id}}],
                        "must_not": [{"terms": {"chunk_id": new_chunk_ids}}],
                    }
                }
            },
            params={"conflicts": "proceed", "refresh": "true"},
        )

    async def delete_chunks(self, document_id: str) -> None:
        await self.initialize()
        await asyncio.to_thread(
            self._client.delete_by_query,
            index=self.index_name,
            body={"query": {"term": {"document_id": document_id}}},
            params={"conflicts": "proceed", "refresh": "true"},
        )

    async def search_chunks(
        self,
        query: str,
        tenant_id: str,
        space_id: str,
        user_subjects: set[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not user_subjects:
            return []
        await self.initialize()
        response = await asyncio.to_thread(
            self._client.search,
            index=self.index_name,
            body={
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["title^2", "text"],
                                    "type": "best_fields",
                                    "minimum_should_match": "1",
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"tenant_id": tenant_id}},
                            {"term": {"space_id": space_id}},
                            {"terms": {"allowed_subjects": sorted(user_subjects)}},
                        ],
                    }
                },
            },
        )
        results = [
            self._result_from_hit(hit)
            for hit in response.get("hits", {}).get("hits", [])
        ]
        print("=============OpenSearch BM25 分数=============")
        for rank, result in enumerate(results, start=1):
            print(
                f"rank={rank} bm25_score={result.score:.4f} "
                f"chunk_id={result.chunk.id} source_uri={result.chunk.source_uri}"
            )
        print("=============================================")
        return results

    async def close(self) -> None:
        if self._client is not None:
            await asyncio.to_thread(self._client.close)

    def _initialize_sync(self) -> None:
        if self._client is None:
            self._client = OpenSearch(hosts=[self.url])
        if not self._client.indices.exists(index=self.index_name):
            self._client.indices.create(index=self.index_name, body=self._index_mapping())

    @staticmethod
    def _index_mapping() -> dict[str, Any]:
        return {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "tenant_id": {"type": "keyword"},
                    "space_id": {"type": "keyword"},
                    "allowed_subjects": {"type": "keyword"},
                    "source_uri": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "ordinal": {"type": "integer"},
                    "version": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        }

    def _index_action(self, chunk: Chunk) -> dict[str, Any]:
        return {
            "_op_type": "index",
            "_index": self.index_name,
            "_id": chunk.id,
            "_source": {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "text": chunk.text,
                "tenant_id": chunk.acl.tenant_id,
                "space_id": chunk.acl.space_id,
                "allowed_subjects": sorted(chunk.acl.allowed_subjects),
                "source_uri": chunk.source_uri,
                "page": chunk.page if chunk.page is not None else -1,
                "ordinal": chunk.ordinal,
                "version": chunk.version,
                "metadata": chunk.metadata,
            },
        }

    @staticmethod
    def _result_from_hit(hit: dict[str, Any]) -> RetrievalResult:
        source = hit["_source"]
        page = source.get("page", -1)
        chunk = Chunk(
            id=source["chunk_id"],
            document_id=source["document_id"],
            text=source["text"],
            acl=ACL(
                tenant_id=source["tenant_id"],
                space_id=source["space_id"],
                allowed_subjects=set(source["allowed_subjects"]),
            ),
            source_uri=source["source_uri"],
            title=source["title"],
            page=None if page == -1 else page,
            ordinal=source["ordinal"],
            version=source["version"],
            metadata=source.get("metadata", {}),
        )
        return RetrievalResult(chunk=chunk, score=float(hit["_score"]), source="opensearch-bm25")


class OpenSearchRetriever(Retriever):
    """OpenSearch BM25 关键词召回端口。"""

    def __init__(self, store: OpenSearchChunkStore) -> None:
        self.store = store

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        space_id: str,
        user_subjects: set[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        return await self.store.search_chunks(query, tenant_id, space_id, user_subjects, top_k)


class HybridRetriever(Retriever):
    """并行执行向量和 BM25 召回，并使用 RRF 融合去重。"""

    def __init__(
        self,
        vector_retriever: Retriever,
        keyword_retriever: Retriever,
        candidate_top_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.candidate_top_k = candidate_top_k
        self.rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        space_id: str,
        user_subjects: set[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        candidate_top_k = max(top_k, self.candidate_top_k)
        vector_results, keyword_results = await asyncio.gather(
            self.vector_retriever.retrieve(
                query, tenant_id, space_id, user_subjects, candidate_top_k
            ),
            self.keyword_retriever.retrieve(
                query, tenant_id, space_id, user_subjects, candidate_top_k
            ),
        )
        return self._rrf([vector_results, keyword_results], top_k)

    def _rrf(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        fused: dict[str, tuple[RetrievalResult, float]] = {}
        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                score = 1 / (self.rrf_k + rank)
                current = fused.get(result.chunk.id)
                if current is None:
                    fused[result.chunk.id] = (result, score)
                else:
                    fused[result.chunk.id] = (current[0], current[1] + score)

        merged = [
            RetrievalResult(chunk=result.chunk, score=score, source="hybrid-rrf")
            for result, score in fused.values()
        ]
        merged.sort(key=lambda item: item.score, reverse=True)
        print("=============混合 RRF 分数=============")
        for rank, result in enumerate(merged[:top_k], start=1):
            print(
                f"rank={rank} rrf_score={result.score:.4f} "
                f"chunk_id={result.chunk.id} source_uri={result.chunk.source_uri}"
            )
        print("======================================")
        return merged[:top_k]
