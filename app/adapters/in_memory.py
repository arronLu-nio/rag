import math
import re
from collections import Counter

from app.domain import (
    AnswerTrace,
    Chunk,
    Citation,
    Document,
    QAAnswer,
    RetrievalResult,
)
from app.ports.contracts import ChatModel, DocumentStore, EmbeddingModel, Reranker, Retriever

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """非常轻量的中英文 tokenizer。

    英文按词，中文额外生成单字和 bigram，让“年假”能匹配到“年假申请”。
    生产环境建议换成 OpenSearch/Elasticsearch 的中文分词器。
    """

    tokens: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        tokens.append(lowered)
        cjk_chars = CJK_RE.findall(lowered)
        if cjk_chars:
            tokens.extend(cjk_chars)
            tokens.extend(
                "".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1)
            )
    return tokens


class InMemoryDocumentStore(DocumentStore):
    """本地开发用的内存存储。

    服务重启后数据会消失。它的作用是先跑通架构，后续再换真实数据库。
    """

    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.chunks: dict[str, Chunk] = {}

    async def save_document(self, document: Document) -> Document:
        self.documents[document.id] = document
        return document

    async def save_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk

    async def get_document(self, document_id: str) -> Document | None:
        return self.documents.get(document_id)


class HashEmbeddingModel(EmbeddingModel):
    """可测试的假 embedding。

    它不是语义模型，只是把 token hash 到固定维度向量，方便不接外部 API 也能跑通流程。
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            vector[hash(token) % self.dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class HybridInMemoryRetriever(Retriever):
    """内存版混合检索。

    同时计算 dense_score 和 sparse_score，用来模拟“向量召回 + 关键词召回”的生产策略。
    """

    def __init__(self, store: InMemoryDocumentStore, embedding_model: EmbeddingModel) -> None:
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
        query_terms = Counter(tokenize(query))
        results: list[RetrievalResult] = []

        for chunk in self.store.chunks.values():
            # 权限过滤必须发生在召回阶段，避免无权限 chunk 进入后续 rerank 或生成。
            if chunk.acl.tenant_id != tenant_id or chunk.acl.space_id != space_id:
                continue
            if not chunk.acl.allows(user_subjects):
                continue
            dense_score = cosine_similarity(query_embedding, chunk.embedding)
            sparse_score = keyword_overlap(query_terms, Counter(tokenize(chunk.text)))
            score = 0.7 * dense_score + 0.3 * sparse_score
            if score > 0:
                results.append(RetrievalResult(chunk=chunk, score=score, source="hybrid-memory"))

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


class SimpleReranker(Reranker):
    """本地假 reranker。

    真实生产里可以替换成 BGE reranker、Jina reranker 或供应商 rerank API。
    """

    async def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        query_tokens = set(tokenize(query))
        reranked: list[RetrievalResult] = []
        for result in results:
            chunk_tokens = set(tokenize(result.chunk.text))
            overlap = len(query_tokens.intersection(chunk_tokens))
            result.rerank_score = result.score + overlap * 0.05
            reranked.append(result)
        return sorted(reranked, key=lambda item: item.rerank_score or item.score, reverse=True)


class GroundedStubChatModel(ChatModel):
    """本地假生成模型。

    它只拼接已授权上下文，不调用大模型。下一步接真实模型时替换这个类即可。
    """

    name = "grounded-stub-model"

    async def answer(self, query: str, contexts: list[RetrievalResult]) -> QAAnswer:
        if not contexts:
            # 没有授权上下文时拒答，避免模型凭空编造答案。
            return QAAnswer(
                answer="未在当前用户有权限访问的知识库中找到可靠依据，暂不回答。",
                citations=[],
                confidence=0.0,
                trace=AnswerTrace(
                    query=query,
                    rewritten_query=query,
                    user_subjects=set(),
                    retrieval_count=0,
                    reranked_count=0,
                    used_model=self.name,
                    refusal_reason="no_authorized_context",
                ),
            )

        top_contexts = contexts[:3]
        citations = [
            Citation(
                document_id=result.chunk.document_id,
                chunk_id=result.chunk.id,
                title=result.chunk.title,
                source_uri=result.chunk.source_uri,
                page=result.chunk.page,
                quote=result.chunk.text[:180],
            )
            for result in top_contexts
        ]
        answer = "根据已授权知识库内容：\n" + "\n".join(
            f"- {result.chunk.text[:220]}" for result in top_contexts
        )
        return QAAnswer(
            answer=answer,
            citations=citations,
            confidence=min(0.95, max(result.score for result in top_contexts)),
            trace=AnswerTrace(
                query=query,
                rewritten_query=query,
                user_subjects=set(),
                retrieval_count=len(contexts),
                reranked_count=len(top_contexts),
                used_model=self.name,
                retrieved_chunk_ids=[result.chunk.id for result in contexts],
            ),
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个已归一化向量的相似度。"""

    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def keyword_overlap(query_terms: Counter[str], chunk_terms: Counter[str]) -> float:
    """计算 query 词在 chunk 中的命中比例。"""

    if not query_terms:
        return 0.0
    overlap = sum(min(count, chunk_terms[token]) for token, count in query_terms.items())
    return overlap / sum(query_terms.values())
