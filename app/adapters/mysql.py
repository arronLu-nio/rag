"""MySQL document metadata storage adapter."""

import re
from datetime import UTC
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, MetaData, String, Table, Text, select, text
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.schema import Column

from app.domain import ACL, Chunk, Document, DocumentStatus
from app.ports.contracts import ChunkStore, DocumentStore, IndexStore

metadata = MetaData()
documents = Table(
    "documents",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("title", String(1024), nullable=False),
    Column("source_uri", String(2048), nullable=False),
    Column("content", Text, nullable=False),
    Column("tenant_id", String(256), nullable=False),
    Column("space_id", String(256), nullable=False),
    Column("allowed_subjects", JSON, nullable=False),
    Column("document_metadata", JSON, nullable=False),
    Column("version", BigInteger, nullable=False),
    Column("status", String(32), nullable=False),
    Column("created_at", DateTime, nullable=False),
)


class MySQLDocumentStore(DocumentStore):
    """在 MySQL 保存完整文档、状态、版本和权限元数据。"""

    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_]+", database):
            raise ValueError(
                "mysql database name may only contain letters, digits, and underscores"
            )
        self.database = database
        self._admin_url = URL.create(
            "mysql+aiomysql",
            username=username,
            password=password,
            host=host,
            port=port,
            database="mysql",
        )
        self._database_url = self._admin_url.set(database=database)
        self._engine: AsyncEngine | None = None

    async def initialize(self) -> None:
        """首次启动时创建数据库和 documents 表。"""

        if self._engine is not None:
            return
        admin_engine = create_async_engine(self._admin_url, pool_pre_ping=True)
        try:
            async with admin_engine.begin() as connection:
                await connection.execute(
                    text(
                        f"CREATE DATABASE IF NOT EXISTS `{self.database}` "
                        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                    )
                )
        finally:
            await admin_engine.dispose()

        self._engine = create_async_engine(self._database_url, pool_pre_ping=True)
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def save_document(self, document: Document) -> Document:
        await self.initialize()
        row = self._row(document)
        statement = insert(documents).values(**row)
        update_values = {
            column: statement.inserted[column]
            for column in row
            if column not in {"id", "created_at"}
        }
        async with self._engine.begin() as connection:
            await connection.execute(statement.on_duplicate_key_update(**update_values))
        return document

    async def get_document(self, document_id: str) -> Document | None:
        await self.initialize()
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(documents).where(documents.c.id == document_id)
            )
            row = result.mappings().first()
        if row is None:
            return None
        return self._document_from_row(row)

    async def list_documents(self, tenant_id: str, space_id: str) -> list[Document]:
        await self.initialize()
        async with self._engine.connect() as connection:
            result = await connection.execute(
                select(documents)
                .where(documents.c.tenant_id == tenant_id)
                .where(documents.c.space_id == space_id)
                .where(documents.c.status != DocumentStatus.DELETED.value)
                .order_by(documents.c.created_at.desc())
            )
            rows = result.mappings().all()
        return [self._document_from_row(row) for row in rows]

    @staticmethod
    def _document_from_row(row: Any) -> Document:
        created_at = row["created_at"]
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        return Document(
            id=row["id"],
            title=row["title"],
            source_uri=row["source_uri"],
            content=row["content"],
            acl=ACL(
                tenant_id=row["tenant_id"],
                space_id=row["space_id"],
                allowed_subjects=set(row["allowed_subjects"]),
            ),
            metadata=row["document_metadata"],
            version=row["version"],
            status=DocumentStatus(row["status"]),
            created_at=created_at,
        )

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    @staticmethod
    def _row(document: Document) -> dict[str, Any]:
        return {
            "id": document.id,
            "title": document.title,
            "source_uri": document.source_uri,
            "content": document.content,
            "tenant_id": document.acl.tenant_id,
            "space_id": document.acl.space_id,
            "allowed_subjects": sorted(document.acl.allowed_subjects),
            "document_metadata": document.metadata,
            "version": document.version,
            "status": document.status.value,
            "created_at": document.created_at.replace(tzinfo=None),
        }


class CompositeDocumentStore(IndexStore):
    """把完整 Document 写入 MySQL，把 Chunk 分发给向量和关键词索引。"""

    def __init__(
        self,
        document_store: MySQLDocumentStore,
        chunk_store: ChunkStore,
        keyword_store: ChunkStore | None = None,
    ) -> None:
        self.document_store = document_store
        self.chunk_store = chunk_store
        self.keyword_store = keyword_store

    async def initialize(self) -> None:
        await self.document_store.initialize()
        await self.chunk_store.initialize()
        if self.keyword_store is not None:
            await self.keyword_store.initialize()

    async def save_document(self, document: Document) -> Document:
        return await self.document_store.save_document(document)

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        await self.chunk_store.save_chunks(chunks)
        if self.keyword_store is not None:
            await self.keyword_store.save_chunks(chunks)

    async def get_document(self, document_id: str) -> Document | None:
        return await self.document_store.get_document(document_id)

    async def list_documents(self, tenant_id: str, space_id: str) -> list[Document]:
        return await self.document_store.list_documents(tenant_id, space_id)

    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        await self.chunk_store.replace_chunks(document_id, chunks)
        if self.keyword_store is not None:
            await self.keyword_store.replace_chunks(document_id, chunks)

    async def delete_chunks(self, document_id: str) -> None:
        await self.chunk_store.delete_chunks(document_id)
        if self.keyword_store is not None:
            await self.keyword_store.delete_chunks(document_id)

    async def close(self) -> None:
        await self.document_store.close()
        await self.chunk_store.close()
        if self.keyword_store is not None:
            await self.keyword_store.close()
