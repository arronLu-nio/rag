import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import quote
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.adapters.local import GroundedStubChatModel
from app.adapters.milvus import MilvusDocumentStore, MilvusRetriever
from app.adapters.mysql import CompositeDocumentStore, MySQLDocumentStore
from app.adapters.opensearch import HybridRetriever, OpenSearchChunkStore, OpenSearchRetriever
from app.domain import ACL, Document, DocumentStatus, QAAnswer
from app.parsers import TextDocumentParser
from app.parsers.text import DocumentParseError
from app.ports.contracts import ChatModel, EmbeddingModel, Reranker
from app.settings import settings
from app.workflows import build_indexing_graph, build_qa_graph


class AppState:
    """应用运行时依赖容器。

    FastAPI 启动时创建一次。完整文档写入 MySQL，Chunk 和向量写入 Milvus。
    """

    def __init__(self) -> None:
        self.document_store = build_document_store()
        self.vector_store = build_vector_store()
        self.keyword_store = build_keyword_store()
        self.store = CompositeDocumentStore(
            self.document_store,
            self.vector_store,
            self.keyword_store,
        )
        self.embedding_model = build_embedding_model()
        self.retriever = HybridRetriever(
            MilvusRetriever(self.vector_store, self.embedding_model),
            OpenSearchRetriever(self.keyword_store),
            candidate_top_k=settings.hybrid_candidate_top_k,
            rrf_k=settings.hybrid_rrf_k,
        )
        self.reranker = build_reranker()
        self.chat_model = build_chat_model()
        self.indexing_graph = build_indexing_graph(self.store, self.embedding_model)
        self.qa_graph = build_qa_graph(
            self.retriever,
            self.reranker,
            self.chat_model,
            min_retrieval_score=settings.min_retrieval_score,
            min_rerank_score=settings.min_rerank_score,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag = AppState()
    await app.state.rag.store.initialize()
    try:
        yield
    finally:
        await app.state.rag.store.close()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
WEB_PAGE = Path(__file__).with_name("static") / "index.html"


def build_embedding_model() -> EmbeddingModel:
    """按配置选择 embedding 模型。

    embedding 是生产检索的前置条件，未配置时直接报错，避免写入无效向量。
    """

    if settings.embedding_provider.lower() == "dashscope" and settings.embedding_api_key:
        from app.adapters.dashscope import DashScopeEmbeddingModel

        return DashScopeEmbeddingModel(
            api_key=settings.embedding_api_key,
            api_base=settings.embedding_api_base,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    raise RuntimeError(
        "DashScope embedding is not configured. Set RAG_EMBEDDING_API_KEY in .env before startup."
    )


def build_document_store() -> MySQLDocumentStore:
    """创建 MySQL 文档存储。"""

    return MySQLDocumentStore(
        host=settings.mysql_host,
        port=settings.mysql_port,
        username=settings.mysql_username,
        password=settings.mysql_password,
        database=settings.mysql_database,
    )


def build_vector_store() -> MilvusDocumentStore:
    """创建 Milvus Chunk 和向量存储。"""

    if settings.vector_provider.lower() != "milvus":
        raise RuntimeError(f"unsupported vector provider: {settings.vector_provider}")
    return MilvusDocumentStore(
        uri=settings.milvus_uri,
        embedding_dimensions=settings.embedding_dimensions,
        documents_collection=settings.milvus_documents_collection,
        chunks_collection=settings.milvus_chunks_collection,
        embedding_version=settings.embedding_version,
    )


def build_keyword_store() -> OpenSearchChunkStore:
    """创建 OpenSearch BM25 关键词索引。"""

    return OpenSearchChunkStore(
        url=settings.opensearch_url,
        index_name=settings.opensearch_index,
    )


def build_reranker() -> Reranker:
    """按配置选择百炼 rerank 模型。"""

    api_key = settings.reranker_api_key or settings.embedding_api_key
    if settings.reranker_provider.lower() == "dashscope" and api_key:
        from app.adapters.dashscope_reranker import DashScopeReranker

        return DashScopeReranker(
            api_key=api_key,
            api_url=settings.reranker_api_url,
            model=settings.reranker_model,
            timeout_seconds=settings.reranker_timeout_seconds,
        )
    raise RuntimeError("DashScope reranker is not configured. Set RAG_RERANKER_API_KEY in .env.")


def build_chat_model() -> ChatModel:
    """按配置选择生成模型。

    本地没有填写 DeepSeek key 时，自动回退到假模型，避免服务启动失败。
    """

    if settings.llm_provider.lower() == "deepseek" and settings.deepseek_api_key:
        from app.adapters.deepseek import DeepSeekChatModel

        return DeepSeekChatModel(
            api_key=settings.deepseek_api_key,
            api_base=settings.deepseek_api_base,
            model=settings.deepseek_model,
        )
    return GroundedStubChatModel()


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


class DocumentSummary(BaseModel):
    """文档列表展示需要的轻量字段，不返回完整正文。"""

    document_id: str
    title: str
    source_uri: str
    version: int
    status: str
    created_at: str


class DocumentDetail(BaseModel):
    """编辑页面读取的完整文档。"""

    document_id: str
    title: str
    source_uri: str
    content: str
    tenant_id: str
    space_id: str
    allowed_subjects: set[str]
    metadata: dict[str, str]
    version: int
    status: str


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


@app.get("/", include_in_schema=False)
async def index_page() -> FileResponse:
    """返回本地知识库操作页。"""

    return FileResponse(WEB_PAGE)


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


@app.post("/documents/upload", response_model=IngestDocumentResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF、DOCX 或 Markdown 文件")],
    tenant_id: Annotated[str, Form()],
    space_id: Annotated[str, Form()],
    allowed_subjects: Annotated[str, Form()],
    state: Annotated[AppState, Depends(get_state)],
    title: Annotated[str | None, Form()] = None,
) -> IngestDocumentResponse:
    """上传可提取文字的文件并复用现有索引流程。

    第一版只支持原生文字层；扫描 PDF 和图片文字不会调用 OCR。
    """

    filename = Path(file.filename or "").name
    parser = TextDocumentParser()
    if not filename or Path(filename).suffix.lower() not in parser.supported_extensions:
        raise HTTPException(
            status_code=415,
            detail="only .pdf, .docx, .md, and .markdown are supported",
        )

    content = await file.read(settings.upload_max_bytes + 1)
    if len(content) > settings.upload_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"file exceeds the {settings.upload_max_bytes // 1024 // 1024} MB upload limit",
        )
    try:
        parsed = await asyncio.to_thread(parser.parse, filename, content)
    except DocumentParseError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    document = Document(
        id=str(uuid4()),
        title=title.strip() if title and title.strip() else Path(filename).stem,
        source_uri=f"upload://{quote(filename)}",
        content=parsed.content,
        acl=ACL(
            tenant_id=tenant_id,
            space_id=space_id,
            allowed_subjects=parse_subjects(allowed_subjects),
        ),
        metadata={
            "filename": filename,
            "content_type": file.content_type or "",
            "parser": parsed.parser,
            "page_count": parsed.page_count,
        },
    )
    document = document.model_copy(
        update={"source_uri": f"upload://{document.id}/{quote(filename)}"}
    )
    saved_path = Path(settings.upload_directory) / document.id / filename
    await asyncio.to_thread(save_uploaded_file, saved_path, content)
    try:
        result = await state.indexing_graph.ainvoke({"document": document})
    except Exception:
        await asyncio.to_thread(remove_uploaded_file, saved_path)
        raise

    indexed = result["document"]
    return IngestDocumentResponse(
        document_id=indexed.id,
        status=indexed.status,
        chunk_count=len(result.get("chunks", [])),
    )


@app.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    tenant_id: str,
    space_id: str,
    state: Annotated[AppState, Depends(get_state)],
) -> list[DocumentSummary]:
    """按租户和知识空间读取可管理的文档列表。"""

    documents = await state.store.list_documents(tenant_id, space_id)
    return [to_document_summary(document) for document in documents]


@app.get("/documents/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    tenant_id: str,
    space_id: str,
    state: Annotated[AppState, Depends(get_state)],
) -> DocumentDetail:
    """读取一篇完整文档，供编辑页面回填。"""

    document = await get_scoped_document(state, document_id, tenant_id, space_id)
    return to_document_detail(document)


@app.put("/documents/{document_id}", response_model=IngestDocumentResponse)
async def update_document(
    document_id: str,
    request: IngestDocumentRequest,
    state: Annotated[AppState, Depends(get_state)],
) -> IngestDocumentResponse:
    """更新全文，并替换该 document_id 在 Milvus 中的全部 Chunk 向量。"""

    current = await get_scoped_document(
        state,
        document_id,
        request.tenant_id,
        request.space_id,
    )
    if current.status == DocumentStatus.DELETED:
        raise HTTPException(status_code=409, detail="document has been deleted")
    document = current.model_copy(
        update={
            "title": request.title,
            "source_uri": request.source_uri,
            "content": request.content,
            "acl": ACL(
                tenant_id=request.tenant_id,
                space_id=request.space_id,
                allowed_subjects=request.allowed_subjects,
            ),
            "metadata": dict(request.metadata),
            "version": current.version + 1,
            "status": DocumentStatus.RECEIVED,
        }
    )
    result = await state.indexing_graph.ainvoke(
        {"document": document, "replace_existing": True}
    )
    indexed = result["document"]
    return IngestDocumentResponse(
        document_id=indexed.id,
        status=indexed.status,
        chunk_count=len(result.get("chunks", [])),
    )


@app.delete("/documents/{document_id}", response_model=DocumentSummary)
async def delete_document(
    document_id: str,
    tenant_id: str,
    space_id: str,
    state: Annotated[AppState, Depends(get_state)],
) -> DocumentSummary:
    """删除 Milvus 向量，并在 MySQL 中保留一条 deleted 状态的审计记录。"""

    current = await get_scoped_document(state, document_id, tenant_id, space_id)
    await state.store.delete_chunks(document_id)
    deleted = current.model_copy(update={"status": DocumentStatus.DELETED})
    await state.store.save_document(deleted)
    return to_document_summary(deleted)


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


async def get_scoped_document(
    state: AppState,
    document_id: str,
    tenant_id: str,
    space_id: str,
) -> Document:
    document = await state.store.get_document(document_id)
    if (
        document is None
        or document.acl.tenant_id != tenant_id
        or document.acl.space_id != space_id
    ):
        raise HTTPException(status_code=404, detail="document not found")
    return document


def to_document_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.id,
        title=document.title,
        source_uri=document.source_uri,
        version=document.version,
        status=document.status,
        created_at=document.created_at.isoformat(),
    )


def to_document_detail(document: Document) -> DocumentDetail:
    return DocumentDetail(
        document_id=document.id,
        title=document.title,
        source_uri=document.source_uri,
        content=document.content,
        tenant_id=document.acl.tenant_id,
        space_id=document.acl.space_id,
        allowed_subjects=document.acl.allowed_subjects,
        metadata=document.metadata,
        version=document.version,
        status=document.status,
    )


def parse_subjects(value: str) -> set[str]:
    subjects = {item.strip() for item in value.split(",") if item.strip()}
    if not subjects:
        raise HTTPException(status_code=422, detail="allowed_subjects is required")
    return subjects


def save_uploaded_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def remove_uploaded_file(path: Path) -> None:
    if path.exists():
        path.unlink()
    if path.parent.exists() and not any(path.parent.iterdir()):
        path.parent.rmdir()
