"""Milvus document storage and vector retrieval adapters."""

import asyncio
import json
from typing import Any

from app.domain import ACL, Chunk, Document, DocumentStatus, RetrievalResult
from app.ports.contracts import EmbeddingModel, IndexStore, Retriever


class MilvusDocumentStore(IndexStore):
    """使用 Milvus 持久化文档元数据和 chunk 向量。

    `documents` collection 保存文档状态和元数据；`chunks` collection 保存可检索文本、
    ACL 和 embedding。Milvus 要求每个 collection 都有向量字段，因此 documents 使用一个
    固定的 marker 向量；它不参与任何检索。Milvus 客户端是同步 SDK，因此通过
    `asyncio.to_thread` 避免阻塞 FastAPI 的事件循环。
    """

    def __init__(
        self,
        uri: str,
        embedding_dimensions: int,
        documents_collection: str,
        chunks_collection: str,
        embedding_version: str,
        client: Any | None = None,
    ) -> None:
        self.uri = uri
        self.embedding_dimensions = embedding_dimensions
        self.documents_collection = documents_collection
        self.chunks_collection = chunks_collection
        self.embedding_version = embedding_version
        self._client = client
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """连接 Milvus，并在首次启动时创建所需 collection。"""

        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._initialize_sync)
            self._initialized = True

    async def close(self) -> None:
        """关闭 Milvus 客户端连接。"""

        if self._client is not None and hasattr(self._client, "close"):
            await asyncio.to_thread(self._client.close)
        self._client = None
        self._initialized = False

    async def save_document(self, document: Document) -> Document:
        await self.initialize()
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self.documents_collection,
            data=[self._document_row(document)],
        )
        return document

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        await self.initialize()
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self.chunks_collection,
            data=[self._chunk_row(chunk) for chunk in chunks],
        )

    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        """写入新版本 Chunk 后，删除同一文档的旧 Chunk。"""

        await self.initialize()
        if not chunks:
            await self.delete_chunks(document_id)
            return
        new_chunk_ids = [chunk.id for chunk in chunks]
        await asyncio.to_thread(
            self._client.upsert,
            collection_name=self.chunks_collection,
            data=[self._chunk_row(chunk) for chunk in chunks],
        )
        ids = json.dumps(new_chunk_ids)
        await asyncio.to_thread(
            self._delete_and_flush,
            (
                f"document_id == {json.dumps(document_id)} and "
                f"id not in {ids}"
            ),
        )

    async def delete_chunks(self, document_id: str) -> None:
        """删除一篇文档对应的全部 Milvus Chunk 向量。"""

        await self.initialize()
        await asyncio.to_thread(
            self._delete_and_flush,
            f"document_id == {json.dumps(document_id)}",
        )

    def _delete_and_flush(self, filter_expression: str) -> None:
        """Milvus 删除后 flush，确保旧 Chunk 不会继续出现在后续检索中。"""

        self._client.delete(
            collection_name=self.chunks_collection,
            filter=filter_expression,
        )
        self._client.flush(collection_name=self.chunks_collection)

    async def get_document(self, document_id: str) -> Document | None:
        await self.initialize()
        rows = await asyncio.to_thread(
            self._client.query,
            collection_name=self.documents_collection,
            filter=f'id == {json.dumps(document_id)}',
            output_fields=[
                "title",
                "source_uri",
                "content",
                "tenant_id",
                "space_id",
                "allowed_subjects",
                "metadata",
                "version",
                "status",
                "created_at",
            ],
        )
        if not rows:
            return None
        return self._document_from_row(document_id, rows[0])

    async def list_documents(self, tenant_id: str, space_id: str) -> list[Document]:
        """文档管理已迁移到 MySQL，Milvus 不提供文档列表。"""

        raise RuntimeError("MilvusDocumentStore does not list documents; use MySQLDocumentStore")

    async def search_chunks(
        self,
        query_embedding: list[float],
        tenant_id: str,
        space_id: str,
        user_subjects: set[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        """在 Milvus 端同时完成租户、空间和 ACL 过滤后再做向量检索。"""

        if not user_subjects:
            return []
        await self.initialize()
        rows = await asyncio.to_thread(
            self._client.search,
            collection_name=self.chunks_collection,
            data=[query_embedding],
            filter=self._access_filter(tenant_id, space_id, user_subjects),
            limit=top_k,
            output_fields=[
                "document_id",
                "text",
                "tenant_id",
                "space_id",
                "allowed_subjects",
                "source_uri",
                "title",
                "page",
                "ordinal",
                "version",
                "metadata",
                "embedding_version",
            ],
            anns_field="embedding",
            search_params={"metric_type": "COSINE"},
        )
        hits = rows[0] if rows else []
        results = [self._result_from_hit(hit) for hit in hits]
        print("=============Milvus 召回分数=============")
        if not results:
            print("没有召回到有权限的 chunk")
        for index, result in enumerate(results, start=1):
            print(
                f"rank={index} score={result.score:.4f} "
                f"chunk_id={result.chunk.id} source_uri={result.chunk.source_uri}"
            )
        print("========================================")
        return results

    def _initialize_sync(self) -> None:
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.uri)

        if not self._client.has_collection(collection_name=self.documents_collection):
            self._client.create_collection(
                collection_name=self.documents_collection,
                schema=self._document_schema(),
            )

        if not self._client.has_collection(collection_name=self.chunks_collection):
            from pymilvus import MilvusClient

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            self._client.create_collection(
                collection_name=self.chunks_collection,
                schema=self._chunk_schema(),
                index_params=index_params,
            )

    def _document_schema(self) -> Any:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="source_uri", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="space_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="allowed_subjects", datatype=DataType.JSON)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="version", datatype=DataType.INT64)
        schema.add_field(field_name="status", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="document_marker", datatype=DataType.FLOAT_VECTOR, dim=2)
        return schema

    def _chunk_schema(self) -> Any:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="tenant_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="space_id", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="allowed_subjects", datatype=DataType.JSON)
        schema.add_field(field_name="source_uri", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="page", datatype=DataType.INT64)
        schema.add_field(field_name="ordinal", datatype=DataType.INT64)
        schema.add_field(field_name="version", datatype=DataType.INT64)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        schema.add_field(field_name="embedding_version", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.embedding_dimensions,
        )
        return schema

    def _document_row(self, document: Document) -> dict[str, Any]:
        return {
            "id": document.id,
            "title": document.title,
            "source_uri": document.source_uri,
            "content": document.content,
            "tenant_id": document.acl.tenant_id,
            "space_id": document.acl.space_id,
            "allowed_subjects": sorted(document.acl.allowed_subjects),
            "metadata": document.metadata,
            "version": document.version,
            "status": document.status.value,
            "created_at": document.created_at.isoformat(),
            # Milvus collection 必须包含向量字段；文档元数据不参与向量检索，所以固定为 [1.0, 0.0]。
            "document_marker": [1.0, 0.0],
        }

    def _chunk_row(self, chunk: Chunk) -> dict[str, Any]:
        if len(chunk.embedding) != self.embedding_dimensions:
            raise ValueError(
                "chunk embedding dimension does not match "
                f"Milvus schema: expected {self.embedding_dimensions}, got {len(chunk.embedding)}"
            )
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "tenant_id": chunk.acl.tenant_id,
            "space_id": chunk.acl.space_id,
            "allowed_subjects": sorted(chunk.acl.allowed_subjects),
            "source_uri": chunk.source_uri,
            "title": chunk.title,
            "page": chunk.page if chunk.page is not None else -1,
            "ordinal": chunk.ordinal,
            "version": chunk.version,
            "metadata": chunk.metadata,
            "embedding_version": self.embedding_version,
            "embedding": chunk.embedding,
        }

    def _document_from_row(self, document_id: str, row: dict[str, Any]) -> Document:
        return Document(
            id=document_id,
            title=row["title"],
            source_uri=row["source_uri"],
            content=row["content"],
            acl=ACL(
                tenant_id=row["tenant_id"],
                space_id=row["space_id"],
                allowed_subjects=set(row["allowed_subjects"]),
            ),
            metadata=row["metadata"],
            version=row["version"],
            status=DocumentStatus(row["status"]),
            created_at=row["created_at"],
        )

    def _result_from_hit(self, hit: dict[str, Any]) -> RetrievalResult:
        entity = hit.get("entity", hit)
        page = entity.get("page", -1)
        chunk = Chunk(
            id=str(hit.get("id", entity.get("id"))),
            document_id=entity["document_id"],
            text=entity["text"],
            acl=ACL(
                tenant_id=entity["tenant_id"],
                space_id=entity["space_id"],
                allowed_subjects=set(entity["allowed_subjects"]),
            ),
            source_uri=entity["source_uri"],
            title=entity["title"],
            page=None if page == -1 else page,
            ordinal=entity["ordinal"],
            version=entity["version"],
            metadata=entity["metadata"],
        )
        return RetrievalResult(
            chunk=chunk,
            score=float(hit.get("distance", hit.get("score", 0.0))),
            source="milvus-vector",
        )

    @staticmethod
    def _access_filter(tenant_id: str, space_id: str, user_subjects: set[str]) -> str:
        subjects = json.dumps(sorted(user_subjects), ensure_ascii=False)
        return (
            f"tenant_id == {json.dumps(tenant_id)} and "
            f"space_id == {json.dumps(space_id)} and "
            f"json_contains_any(allowed_subjects, {subjects})"
        )


class MilvusRetriever(Retriever):
    """只负责 query embedding 和 Milvus 向量召回。"""

    def __init__(self, store: MilvusDocumentStore, embedding_model: EmbeddingModel) -> None:
        self.store = store
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        space_id: str,
        user_subjects: set[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        query_embedding = (await self.embedding_model.embed([query]))[0]
        return await self.store.search_chunks(
            query_embedding=query_embedding,
            tenant_id=tenant_id,
            space_id=space_id,
            user_subjects=user_subjects,
            top_k=top_k,
        )
