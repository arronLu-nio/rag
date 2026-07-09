# Enterprise Knowledge Base RAG

> A LangGraph-first enterprise RAG scaffold for permission-aware knowledge-base
> question answering.

This project demonstrates the final architecture shape for an enterprise
knowledge-base QA system: documents are ingested, chunked, embedded, retrieved
with ACL filtering, reranked, and answered with citations and traceability.

The current implementation is intentionally local-first. It uses in-memory
adapters and a deterministic stub model so the full workflow can run without
Milvus, OpenSearch, PostgreSQL, Redis, or external LLM credentials. Production
systems can replace those adapters behind stable ports.

## Why This Shape

Enterprise RAG is not just "send chunks to an LLM". It needs a controlled data
contract, reliable permissions, observable workflows, and graceful refusal when
authorized context is missing.

This repository therefore uses:

- **LangGraph** as the workflow runtime for indexing and QA orchestration.
- **LangChain Core** only as an integration toolbox, not as the main business
  model.
- **Custom domain types** for `Document`, `Chunk`, `ACL`, `Citation`,
  `RetrievalResult`, and `AnswerTrace`.
- **Ports and adapters** so local memory components can later become Milvus,
  OpenSearch, PostgreSQL, object storage, and real model providers.

## Target Final Architecture

```text
Documents / Web Pages / Office Files
        |
        v
Ingestion API
        |
        v
LangGraph Indexing Workflow
  receive -> parse -> chunk -> embed -> persist indexes -> publish
        |
        +--> Object Storage: original files
        +--> PostgreSQL: metadata, ACL, versions, traces
        +--> Milvus: dense vectors
        +--> OpenSearch: BM25 / keyword index

User Question
        |
        v
LangGraph QA Workflow
  permission -> rewrite -> hybrid retrieve -> rerank -> generate -> trace
        |
        v
Answer + Citations + Confidence + Audit Trace
```

## Current Capabilities

- Document ingestion through a FastAPI endpoint.
- LangGraph indexing workflow.
- Text chunking with overlap.
- Deterministic local embedding for development.
- Permission-aware retrieval using `tenant_id`, `space_id`, and subject ACLs.
- Hybrid in-memory retrieval that simulates dense vector + keyword search.
- Local reranking stage.
- Grounded stub answer generation with citations.
- Refusal when no authorized context is found.
- Trace output for query, rewritten query, retrieved chunks, model, and refusal
  reason.
- API and workflow tests.

## Project Layout

```text
app/
  main.py                  FastAPI entry point
  settings.py              Environment-based settings
  domain/
    models.py              Business-owned RAG data contracts
  ports/
    contracts.py           Interfaces for storage, retrieval, rerank, models
  adapters/
    in_memory.py           Local development implementations
  workflows/
    indexing.py            LangGraph document indexing graph
    qa.py                  LangGraph question-answering graph
tests/
  test_api.py              HTTP API tests
  test_rag_workflows.py    Workflow and permission tests
```

## Quick Start

Create and activate the conda environment:

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

Install dependencies:

```bash
pip install \
  fastapi \
  "uvicorn[standard]" \
  langgraph \
  langchain-core \
  pydantic \
  pydantic-settings \
  python-multipart \
  pytest \
  pytest-asyncio \
  httpx \
  ruff
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

If your shell is not activated, run through conda:

```bash
conda run -n rag uvicorn app.main:app --reload
```

## API Example

Ingest a document:

```bash
curl -X POST http://127.0.0.1:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "IT制度",
    "source_uri": "manual://it",
    "content": "VPN 账号申请需要直属主管审批。",
    "tenant_id": "t1",
    "space_id": "it",
    "allowed_subjects": ["user:bob"]
  }'
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/qa/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "VPN 账号怎么申请？",
    "tenant_id": "t1",
    "space_id": "it",
    "user_subjects": ["user:bob"],
    "top_k": 8
  }'
```

The response includes:

- `answer`: generated answer
- `citations`: source chunks used as evidence
- `confidence`: simple confidence score
- `trace`: retrieval and model trace for debugging

## Test

```bash
pytest
ruff check .
```

Or without activating the environment:

```bash
conda run -n rag pytest
conda run -n rag ruff check .
```

## Production Roadmap

- Replace `HashEmbeddingModel` with a real embedding provider such as BGE-M3,
  Tongyi, Zhipu, Volcengine, DeepSeek-compatible services, or OpenAI-compatible
  APIs.
- Replace `GroundedStubChatModel` with a multi-provider chat adapter.
- Replace `InMemoryDocumentStore` with PostgreSQL metadata storage plus object
  storage for original files.
- Replace `HybridInMemoryRetriever` with Milvus dense vector retrieval and
  OpenSearch keyword retrieval.
- Add a production reranker such as BGE reranker, Jina reranker, or a cloud
  rerank API.
- Add document parsers for PDF, Word, Excel, Markdown, HTML, and web crawling.
- Add background indexing with Redis and Celery/RQ.
- Add evaluation with Ragas and a curated enterprise QA benchmark set.
- Add admin UI for documents, permissions, indexing status, feedback, and audit
  traces.

## Design Principles

- Keep business contracts independent from framework-specific schemas.
- Apply ACL filtering before rerank and generation.
- Always return citations for grounded answers.
- Refuse when no authorized context exists.
- Make every answer traceable.
- Keep local development lightweight while preserving a production migration
  path.
