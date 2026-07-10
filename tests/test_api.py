from httpx import ASGITransport, AsyncClient

from app.adapters.local import GroundedStubChatModel, SimpleReranker
from app.main import app, build_chat_model
from app.workflows import build_indexing_graph, build_qa_graph
from tests.fakes import FakeDocumentStore, FakeEmbeddingModel, FakeRetriever


class ApiTestState:
    def __init__(self) -> None:
        self.store = FakeDocumentStore()
        self.embedding_model = FakeEmbeddingModel()
        self.retriever = FakeRetriever(self.store, self.embedding_model)
        self.reranker = SimpleReranker()
        self.chat_model = GroundedStubChatModel()
        self.indexing_graph = build_indexing_graph(self.store, self.embedding_model)
        self.qa_graph = build_qa_graph(self.retriever, self.reranker, self.chat_model)


def use_test_state(monkeypatch) -> None:
    monkeypatch.setattr("app.main.AppState", ApiTestState)


async def test_health_endpoint(monkeypatch):
    use_test_state(monkeypatch)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_index_page(monkeypatch):
    use_test_state(monkeypatch)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

    assert response.status_code == 200
    assert "企业知识库" in response.text
    assert "存入知识库" in response.text


async def test_ingest_then_ask_endpoint(monkeypatch):
    use_test_state(monkeypatch)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            ingest_response = await client.post(
                "/documents/ingest",
                json={
                    "title": "IT制度",
                    "source_uri": "manual://it",
                    "content": "VPN 账号申请需要直属主管审批。",
                    "tenant_id": "t1",
                    "space_id": "it",
                    "allowed_subjects": ["user:bob"],
                },
            )
            ask_response = await client.post(
                "/qa/ask",
                json={
                    "query": "VPN 账号怎么申请？",
                    "tenant_id": "t1",
                    "space_id": "it",
                    "user_subjects": ["user:bob"],
                },
            )

    assert ingest_response.status_code == 200
    assert ingest_response.json()["status"] == "published"
    assert ask_response.status_code == 200
    answer = ask_response.json()
    assert answer["citations"]
    assert answer["trace"]["used_model"] == "grounded-stub-model"


async def test_list_update_and_delete_document(monkeypatch):
    use_test_state(monkeypatch)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            create_response = await client.post(
                "/documents/ingest",
                json={
                    "title": "IT制度",
                    "source_uri": "manual://it",
                    "content": "VPN 账号申请需要直属主管审批。",
                    "tenant_id": "t1",
                    "space_id": "it",
                    "allowed_subjects": ["user:bob"],
                },
            )
            document_id = create_response.json()["document_id"]

            list_response = await client.get("/documents?tenant_id=t1&space_id=it")
            detail_response = await client.get(
                f"/documents/{document_id}?tenant_id=t1&space_id=it"
            )
            update_response = await client.put(
                f"/documents/{document_id}",
                json={
                    "title": "新版 IT制度",
                    "source_uri": "manual://it",
                    "content": "VPN 账号申请需要在 IT 服务台提交，并由直属主管审批。",
                    "tenant_id": "t1",
                    "space_id": "it",
                    "allowed_subjects": ["user:bob"],
                },
            )
            updated_detail_response = await client.get(
                f"/documents/{document_id}?tenant_id=t1&space_id=it"
            )
            delete_response = await client.delete(
                f"/documents/{document_id}?tenant_id=t1&space_id=it"
            )
            after_delete_response = await client.get("/documents?tenant_id=t1&space_id=it")

    assert create_response.status_code == 200
    assert list_response.json()[0]["document_id"] == document_id
    assert detail_response.json()["content"] == "VPN 账号申请需要直属主管审批。"
    assert update_response.status_code == 200
    assert update_response.json()["document_id"] == document_id
    assert updated_detail_response.json()["version"] == 2
    assert "IT 服务台" in updated_detail_response.json()["content"]
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert after_delete_response.json() == []


async def test_upload_markdown_document(monkeypatch, tmp_path):
    use_test_state(monkeypatch)
    monkeypatch.setattr("app.main.settings.upload_directory", str(tmp_path))

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/documents/upload",
                data={
                    "tenant_id": "t1",
                    "space_id": "it",
                    "allowed_subjects": "user:bob",
                    "title": "VPN 手册",
                },
                files={
                    "file": (
                        "vpn.md",
                        "# VPN\n\nVPN 账号申请需要主管审批。".encode(),
                        "text/markdown",
                    )
                },
            )

    assert response.status_code == 200
    document_id = response.json()["document_id"]
    stored = app.state.rag.store.documents[document_id]
    assert stored.title == "VPN 手册"
    assert stored.metadata["parser"] == "utf-8"
    assert (tmp_path / document_id / "vpn.md").exists()


def test_empty_deepseek_key_falls_back_to_stub(monkeypatch):
    monkeypatch.setattr("app.main.settings.llm_provider", "deepseek")
    monkeypatch.setattr("app.main.settings.deepseek_api_key", "")

    assert isinstance(build_chat_model(), GroundedStubChatModel)
