from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.adapters.in_memory import (
    GroundedStubChatModel,
    HashEmbeddingModel,
    HybridInMemoryRetriever,
    InMemoryDocumentStore,
    SimpleReranker,
)
from app.domain import ACL, Document, QAAnswer
from app.settings import settings
from app.workflows import build_indexing_graph, build_qa_graph


class AppState:
    """应用运行时依赖容器。

    FastAPI 启动时创建一次。当前使用内存适配器，后续可以在这里切换到真实组件。
    """

    def __init__(self) -> None:
        self.store = InMemoryDocumentStore()
        self.embedding_model = HashEmbeddingModel()
        self.retriever = HybridInMemoryRetriever(self.store, self.embedding_model)
        self.reranker = SimpleReranker()
        self.chat_model = GroundedStubChatModel()
        self.indexing_graph = build_indexing_graph(self.store, self.embedding_model)
        self.qa_graph = build_qa_graph(self.retriever, self.reranker, self.chat_model)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = AppState()
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


def get_state() -> AppState:
    return app.state.rag


class IngestDocumentRequest(BaseModel):
    """文档入库请求。

    这里先收纯文本 content；后续文件上传接口会把 PDF/Word 解析结果转成同样的结构。
    """

    title: str
    source_uri: str = "manual://local"
    content: str = Field(min_length=1)
    tenant_id: str
    space_id: str
    allowed_subjects: set[str]
    metadata: dict[str, str] = Field(default_factory=dict)


class IngestDocumentResponse(BaseModel):
    """文档入库后的简要结果。"""

    document_id: str
    status: str
    chunk_count: int


class AskRequest(BaseModel):
    """问答请求。"""

    query: str = Field(min_length=1)
    tenant_id: str
    space_id: str
    user_subjects: set[str]
    top_k: int = Field(default=8, ge=1, le=50)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "environment": settings.environment}


@app.post("/documents/ingest", response_model=IngestDocumentResponse)
async def ingest_document(
    request: IngestDocumentRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> IngestDocumentResponse:
    """把一篇文档送入 LangGraph 索引流程。"""

    document = Document(
        title=request.title,
        source_uri=request.source_uri,
        content=request.content,
        acl=ACL(
            tenant_id=request.tenant_id,
            space_id=request.space_id,
            allowed_subjects=request.allowed_subjects,
        ),
        metadata=dict(request.metadata),
    )
    result = await state.indexing_graph.ainvoke({"document": document})
    indexed = result["document"]
    return IngestDocumentResponse(
        document_id=indexed.id,
        status=indexed.status,
        chunk_count=len(result.get("chunks", [])),
    )


@app.post("/qa/ask", response_model=QAAnswer)
async def ask(
    request: AskRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> QAAnswer:
    """执行一次带权限过滤和引用追踪的知识库问答。"""

    if not request.user_subjects:
        raise HTTPException(status_code=403, detail="user_subjects is required")
    result = await state.qa_graph.ainvoke(
        {
            "query": request.query,
            "tenant_id": request.tenant_id,
            "space_id": request.space_id,
            "user_subjects": request.user_subjects,
            "top_k": request.top_k,
        }
    )
    return result["answer"]
