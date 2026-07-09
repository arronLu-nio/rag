from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    """文档从接收到可检索发布的生命周期状态。"""

    RECEIVED = "received"
    PARSED = "parsed"
    INDEXED = "indexed"
    PUBLISHED = "published"
    FAILED = "failed"


class ACL(BaseModel):
    """最小权限模型。

    tenant_id 做租户隔离，space_id 做知识库/空间隔离，allowed_subjects 表示谁能看。
    subject 可以是 user:alice、dept:finance、role:admin 这类字符串。
    """

    tenant_id: str
    space_id: str
    allowed_subjects: set[str] = Field(default_factory=set)

    def allows(self, user_subjects: set[str]) -> bool:
        """只要用户身份集合和文档授权集合有交集，就认为有权限。"""

        if not self.allowed_subjects:
            return False
        return bool(self.allowed_subjects.intersection(user_subjects))


class Document(BaseModel):
    """原始文档。

    生产环境里 content 可以来自文件解析结果，source_uri 指向 MinIO/S3/网页/本地上传来源。
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_uri: str
    content: str
    acl: ACL
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    status: DocumentStatus = DocumentStatus.RECEIVED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Chunk(BaseModel):
    """切分后的可检索片段。

    RAG 真正召回的是 chunk，不是整篇 document，所以权限、来源、版本信息也要复制到 chunk 上。
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    acl: ACL
    source_uri: str
    title: str
    page: int | None = None
    ordinal: int = 0
    version: int = 1
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """检索阶段返回的候选片段和分数。"""

    chunk: Chunk
    score: float
    source: str
    rerank_score: float | None = None


class Citation(BaseModel):
    """最终答案引用的来源。

    前端可以用这些字段展示“答案依据来自哪篇文档、哪一段、哪一页”。
    """

    document_id: str
    chunk_id: str
    title: str
    source_uri: str
    page: int | None = None
    quote: str


class AnswerTrace(BaseModel):
    """一次问答的可追踪信息。

    用于排查为什么这么回答、召回了哪些 chunk、是否拒答、用了哪个模型。
    """

    query: str
    rewritten_query: str
    user_subjects: set[str]
    retrieval_count: int
    reranked_count: int
    used_model: str
    refusal_reason: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)


class QAAnswer(BaseModel):
    """问答接口最终返回给调用方的数据结构。"""

    answer: str
    citations: list[Citation]
    confidence: float
    trace: AnswerTrace
