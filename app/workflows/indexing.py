from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.domain import Chunk, Document, DocumentStatus
from app.ports.contracts import EmbeddingModel, IndexStore


class IndexingState(TypedDict, total=False):
    """LangGraph 索引流程里的共享状态。"""

    document: Document
    chunks: list[Chunk]
    replace_existing: bool
    error: str | None


def build_indexing_graph(store: IndexStore, embedding_model: EmbeddingModel):
    """构建文档入库图。

    流程：接收文档 -> 解析切分 -> 生成 embedding -> 持久化索引 -> 发布版本。
    """

    async def receive_document(state: IndexingState) -> IndexingState:
        """保存原始文档，标记为 received。"""

        document = state["document"].model_copy(update={"status": DocumentStatus.RECEIVED})
        await store.save_document(document)
        return {**state, "document": document}

    async def parse_and_chunk(state: IndexingState) -> IndexingState:
        """把文档正文切成 chunk。

        现在只处理纯文本；以后 PDF/Word/HTML 解析器会在这个节点之前或节点内接入。
        """

        document = state["document"].model_copy(update={"status": DocumentStatus.PARSED})
        chunks = [
            Chunk(
                document_id=document.id,
                text=text,
                acl=document.acl,
                source_uri=document.source_uri,
                title=document.title,
                ordinal=index,
                version=document.version,
            )
            for index, text in enumerate(split_text(document.content))
            if text.strip()
        ]
        return {**state, "document": document, "chunks": chunks}

    async def embed_chunks(state: IndexingState) -> IndexingState:
        """为每个 chunk 生成向量。"""

        chunks = state["chunks"]
        embeddings = await embedding_model.embed([chunk.text for chunk in chunks])
        embedded = [
            chunk.model_copy(update={"embedding": embedding})
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        return {**state, "chunks": embedded}

    async def persist_indexes(state: IndexingState) -> IndexingState:
        """保存 chunk 和索引数据。

        当前写入 Milvus：文档状态写入 documents collection，chunk 和向量写入 chunks collection。
        """

        document = state["document"].model_copy(update={"status": DocumentStatus.INDEXED})
        await store.save_document(document)
        if state.get("replace_existing"):
            await store.replace_chunks(document.id, state["chunks"])
        else:
            await store.save_chunks(state["chunks"])
        return {**state, "document": document}

    async def publish_version(state: IndexingState) -> IndexingState:
        """把文档版本标记为可被问答流程使用。"""

        document = state["document"].model_copy(update={"status": DocumentStatus.PUBLISHED})
        await store.save_document(document)
        return {**state, "document": document}

    graph = StateGraph(IndexingState)
    graph.add_node("receive_document", receive_document)
    graph.add_node("parse_and_chunk", parse_and_chunk)
    graph.add_node("embed_chunks", embed_chunks)
    graph.add_node("persist_indexes", persist_indexes)
    graph.add_node("publish_version", publish_version)
    graph.set_entry_point("receive_document")
    graph.add_edge("receive_document", "parse_and_chunk")
    graph.add_edge("parse_and_chunk", "embed_chunks")
    graph.add_edge("embed_chunks", "persist_indexes")
    graph.add_edge("persist_indexes", "publish_version")
    graph.add_edge("publish_version", END)
    return graph.compile()


def split_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """简单字符切分。

    overlap 会让相邻 chunk 保留一点重复内容，降低答案跨边界时召回失败的概率。
    """

    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks
