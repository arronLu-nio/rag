import pytest

from app.adapters.in_memory import (
    GroundedStubChatModel,
    HashEmbeddingModel,
    HybridInMemoryRetriever,
    InMemoryDocumentStore,
    SimpleReranker,
)
from app.domain import ACL, Document, DocumentStatus
from app.workflows import build_indexing_graph, build_qa_graph


@pytest.fixture
def rag_components():
    store = InMemoryDocumentStore()
    embedding_model = HashEmbeddingModel()
    retriever = HybridInMemoryRetriever(store, embedding_model)
    reranker = SimpleReranker()
    chat_model = GroundedStubChatModel()
    return store, build_indexing_graph(store, embedding_model), build_qa_graph(
        retriever, reranker, chat_model
    )


async def test_indexing_publishes_document_and_chunks(rag_components):
    store, indexing_graph, _ = rag_components
    document = Document(
        title="报销制度",
        source_uri="manual://policy",
        content="差旅报销需要在30天内提交发票和审批单。",
        acl=ACL(tenant_id="t1", space_id="hr", allowed_subjects={"dept:finance"}),
    )

    result = await indexing_graph.ainvoke({"document": document})

    assert result["document"].status == DocumentStatus.PUBLISHED
    assert await store.get_document(document.id) is not None
    assert len(store.chunks) == 1


async def test_qa_filters_unauthorized_chunks(rag_components):
    _, indexing_graph, qa_graph = rag_components
    allowed_doc = Document(
        title="公开制度",
        source_uri="manual://public",
        content="年假申请需要提前三天提交。",
        acl=ACL(tenant_id="t1", space_id="hr", allowed_subjects={"user:alice"}),
    )
    denied_doc = Document(
        title="高管制度",
        source_uri="manual://secret",
        content="高管奖金方案包含敏感预算。",
        acl=ACL(tenant_id="t1", space_id="hr", allowed_subjects={"role:executive"}),
    )
    await indexing_graph.ainvoke({"document": allowed_doc})
    await indexing_graph.ainvoke({"document": denied_doc})

    result = await qa_graph.ainvoke(
        {
            "query": "年假怎么申请？高管奖金是多少？",
            "tenant_id": "t1",
            "space_id": "hr",
            "user_subjects": {"user:alice"},
            "top_k": 5,
        }
    )

    answer = result["answer"]
    assert answer.citations
    assert all(citation.source_uri != "manual://secret" for citation in answer.citations)
    assert all("高管奖金" not in citation.quote for citation in answer.citations)


async def test_qa_refuses_when_no_authorized_context(rag_components):
    _, indexing_graph, qa_graph = rag_components
    document = Document(
        title="财务制度",
        source_uri="manual://finance",
        content="采购付款需要合同和验收单。",
        acl=ACL(tenant_id="t1", space_id="finance", allowed_subjects={"dept:finance"}),
    )
    await indexing_graph.ainvoke({"document": document})

    result = await qa_graph.ainvoke(
        {
            "query": "采购付款需要什么？",
            "tenant_id": "t1",
            "space_id": "finance",
            "user_subjects": {"dept:hr"},
            "top_k": 5,
        }
    )

    answer = result["answer"]
    assert answer.citations == []
    assert answer.confidence == 0.0
    assert answer.trace.refusal_reason == "no_authorized_context"


async def test_qa_trace_records_retrieval_and_model(rag_components):
    _, indexing_graph, qa_graph = rag_components
    document = Document(
        title="IT制度",
        source_uri="manual://it",
        content="VPN 账号申请需要直属主管审批。",
        acl=ACL(tenant_id="t1", space_id="it", allowed_subjects={"user:bob"}),
    )
    await indexing_graph.ainvoke({"document": document})

    result = await qa_graph.ainvoke(
        {
            "query": "VPN 账号怎么申请？",
            "tenant_id": "t1",
            "space_id": "it",
            "user_subjects": {"user:bob"},
            "top_k": 5,
        }
    )

    trace = result["answer"].trace
    assert trace.query == "VPN 账号怎么申请？"
    assert trace.retrieval_count >= 1
    assert trace.reranked_count >= 1
    assert trace.used_model == "grounded-stub-model"
    assert trace.retrieved_chunk_ids
