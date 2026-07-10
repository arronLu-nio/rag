from app.adapters.milvus import MilvusDocumentStore
from app.domain import ACL, Chunk, Document, DocumentStatus


class FakeMilvusClient:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created: list[dict] = []
        self.upserts: list[dict] = []
        self.search_calls: list[dict] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs) -> None:
        self.collections.add(kwargs["collection_name"])
        self.created.append(kwargs)

    def upsert(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def query(self, **kwargs):
        return []

    def flush(self, **kwargs) -> None:
        return None

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [
            [
                {
                    "id": "chunk-1",
                    "distance": 0.91,
                    "entity": {
                        "document_id": "doc-1",
                        "text": "VPN 账号申请需要主管审批。",
                        "tenant_id": "t1",
                        "space_id": "it",
                        "allowed_subjects": ["user:bob"],
                        "source_uri": "manual://it",
                        "title": "IT制度",
                        "page": -1,
                        "ordinal": 0,
                        "version": 1,
                        "metadata": {},
                        "embedding_version": "test-v1",
                    },
                }
            ]
        ]


def build_store() -> tuple[MilvusDocumentStore, FakeMilvusClient]:
    client = FakeMilvusClient()
    store = MilvusDocumentStore(
        uri="http://milvus.test:19530",
        embedding_dimensions=3,
        documents_collection="test_documents",
        chunks_collection="test_chunks",
        embedding_version="test-v1",
        client=client,
    )
    return store, client


async def test_milvus_store_upserts_document_and_chunk_with_version():
    store, client = build_store()
    document = Document(
        id="doc-1",
        title="IT制度",
        source_uri="manual://it",
        content="VPN 账号申请需要主管审批。",
        acl=ACL(tenant_id="t1", space_id="it", allowed_subjects={"user:bob"}),
        status=DocumentStatus.PUBLISHED,
    )
    chunk = Chunk(
        id="chunk-1",
        document_id=document.id,
        text=document.content,
        acl=document.acl,
        source_uri=document.source_uri,
        title=document.title,
        embedding=[0.1, 0.2, 0.3],
    )

    await store.save_document(document)
    await store.save_chunks([chunk])

    assert {item["collection_name"] for item in client.created} == {
        "test_documents",
        "test_chunks",
    }
    assert client.upserts[0]["data"][0]["status"] == "published"
    assert client.upserts[1]["data"][0]["embedding_version"] == "test-v1"
    assert client.upserts[1]["data"][0]["embedding"] == [0.1, 0.2, 0.3]


async def test_milvus_search_filters_acl_inside_milvus():
    store, client = build_store()

    results = await store.search_chunks(
        query_embedding=[0.1, 0.2, 0.3],
        tenant_id="t1",
        space_id="it",
        user_subjects={"user:bob", "role:employee"},
        top_k=5,
    )

    search_filter = client.search_calls[0]["filter"]
    assert 'tenant_id == "t1"' in search_filter
    assert 'space_id == "it"' in search_filter
    assert "json_contains_any(allowed_subjects" in search_filter
    assert results[0].source == "milvus-vector"
    assert results[0].chunk.page is None
