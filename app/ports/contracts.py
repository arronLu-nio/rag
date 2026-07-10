from abc import ABC, abstractmethod

from app.domain import Chunk, Document, QAAnswer, RetrievalResult


class DocumentStore(ABC):
    """完整 Document 的存储端口。"""

    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def save_document(self, document: Document) -> Document:
        raise NotImplementedError

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None:
        raise NotImplementedError

    @abstractmethod
    async def list_documents(self, tenant_id: str, space_id: str) -> list[Document]:
        raise NotImplementedError

class ChunkStore(ABC):
    """Chunk 及其检索索引的存储端口。"""

    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def replace_chunks(self, document_id: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete_chunks(self, document_id: str) -> None:
        raise NotImplementedError


class IndexStore(DocumentStore, ChunkStore):
    """同时提供 Document 和 Chunk 能力的入库端口。"""


class EmbeddingModel(ABC):
    """把文本转成向量的端口。

    真实环境可以接 BGE-M3、通义/智谱/火山 embedding，或者任何 OpenAI-compatible API。
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class Retriever(ABC):
    """召回端口，负责按权限找出和 query 相关的 chunk。"""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        space_id: str,
        user_subjects: set[str],
        top_k: int,
    ) -> list[RetrievalResult]:
        raise NotImplementedError


class Reranker(ABC):
    """重排端口，把初召回结果按更精细的相关性重新排序。"""

    @abstractmethod
    async def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        raise NotImplementedError


class ChatModel(ABC):
    """生成答案的模型端口。"""

    name: str

    @abstractmethod
    async def answer(self, query: str, contexts: list[RetrievalResult]) -> QAAnswer:
        raise NotImplementedError
