from abc import ABC, abstractmethod

from app.domain import Chunk, Document, QAAnswer, RetrievalResult


class DocumentStore(ABC):
    """文档和 chunk 的存储端口。

    当前实现是内存版；以后可以替换成 PostgreSQL + Milvus + OpenSearch 的组合。
    """

    @abstractmethod
    async def save_document(self, document: Document) -> Document:
        raise NotImplementedError

    @abstractmethod
    async def save_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_document(self, document_id: str) -> Document | None:
        raise NotImplementedError


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
