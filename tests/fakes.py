"""仅供 pytest 使用的测试替身，不属于应用运行时代码。"""

import math
from collections import Counter

from app.domain import Chunk, Document, RetrievalResult
from app.ports.contracts import EmbeddingModel, IndexStore, Retriever


def _tokens(text: str) -> list[str]:
    return [char.lower() for char in text if not char.isspace()]


class FakeDocumentStore(IndexStore):
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def save_document(self, document: Document) -> Document:
        self.documents[document.id] = document
        return document

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.update({chunk.id: chunk for chunk in chunks})

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)

    async def list_documents(self, tenant_id: str, space_id: str) -> list[Document]:
        return [
            document
            for document in self.documents.values()
            if document.acl.tenant_id == tenant_id
            and document.acl.space_id == space_id
            and document.status.value != "deleted"
        ]

    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        await self.delete_chunks(document_id)
        await self.save_chunks(chunks)

    async def delete_chunks(self, document_id: str) -> None:
        self.chunks = {
            chunk_id: chunk
            for chunk_id, chunk in self.chunks.items()
            if chunk.document_id != document_id
        }


class FakeEmbeddingModel(EmbeddingModel):
    """固定维度的测试向量，避免测试访问外部 embedding 服务。"""

    dimensions = 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in _tokens(text):
                vector[ord(token) % self.dimensions] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


class FakeRetriever(Retriever):
    def __init__(self, store: FakeDocumentStore, embedding_model: EmbeddingModel) -> None:
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
        query_terms = Counter(_tokens(query))
        results: list[RetrievalResult] = []
        for chunk in self.store.chunks.values():
            if chunk.acl.tenant_id != tenant_id or chunk.acl.space_id != space_id:
                continue
            if not chunk.acl.allows(user_subjects):
                continue
            chunk_terms = Counter(_tokens(chunk.text))
            score = sum(min(count, chunk_terms[token]) for token, count in query_terms.items())
            if score:
                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=float(score),
                        source="test-fake",
                        vector_score=float(score),
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
