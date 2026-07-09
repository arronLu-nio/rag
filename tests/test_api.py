from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_health_endpoint():
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ingest_then_ask_endpoint():
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
