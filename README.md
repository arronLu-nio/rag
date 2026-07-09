# 企业知识库 RAG 问答系统

> 基于 **LangGraph 主编排 + LangChain 辅助集成** 的企业级知识库问答骨架。

这是一个面向企业文档问答的 RAG 项目模板。它不是简单的“把文档切块后丢给大模型”，而是从一开始就把 **权限隔离、引用溯源、流程可观测、模型可替换、生产扩展路径** 放进架构里。

当前版本默认使用内存存储、假 embedding 和假生成模型，方便本地快速跑通完整链路；后续可以逐步替换成 Milvus、OpenSearch、PostgreSQL、对象存储和真实大模型服务。

![企业知识库 RAG 架构图](docs/images/architecture.svg)

## 项目定位

这个项目的目标是做一个可演进的企业知识库问答底座：

- 支持企业文档入库、切分、索引、检索、问答。
- 支持租户、知识空间和用户/部门/角色级权限隔离。
- 每次回答都返回引用来源，避免“模型说了但不知道依据在哪”。
- 无授权上下文时明确拒答，避免泄露和胡编。
- 核心业务类型自研，不把 LangChain 的通用类型当作业务模型。
- LangGraph 负责流程编排，LangChain 只作为模型/工具集成层。

## 技术选型

| 模块 | 当前实现 | 生产形态 |
| --- | --- | --- |
| Web API | FastAPI | FastAPI / 网关 / 企业认证 |
| 工作流编排 | LangGraph | LangGraph + 持久化状态 + 重试 |
| 领域模型 | 自定义 Pydantic 类型 | 保持自定义，作为系统核心契约 |
| 文档存储 | 内存 | PostgreSQL + MinIO/S3 |
| 向量检索 | 内存模拟 | Milvus |
| 关键词检索 | 内存模拟 | OpenSearch / Elasticsearch |
| Embedding | Hash 假向量 | BGE-M3 / 通义 / 智谱 / 火山 / OpenAI-compatible API |
| Rerank | 简单词重合 | BGE reranker / Jina reranker / 云厂商 rerank API |
| 生成模型 | GroundedStubChatModel | 多供应商 LLM adapter |
| 测试 | pytest + httpx | 单测、集成测试、RAG 评测集 |

## 核心流程

### 文档入库流程

```text
上传文档
  -> 保存原始文档
  -> 解析正文
  -> 切分 Chunk
  -> 生成 Embedding
  -> 写入索引
  -> 发布版本
```

对应代码：

- `app/workflows/indexing.py`
- `app/domain/models.py`
- `app/adapters/in_memory.py`

### 问答流程

```text
用户提问
  -> 解析权限
  -> Query 改写
  -> 权限过滤召回
  -> 混合检索
  -> Rerank
  -> 生成答案
  -> 返回引用和 Trace
```

对应代码：

- `app/workflows/qa.py`
- `app/main.py`
- `app/ports/contracts.py`

## 为什么不用 LangChain 的 Document 当核心模型

LangChain 的 `Document` 很适合快速 Demo，但企业 RAG 需要更强的业务约束。

本项目自定义了：

- `Document`：原始文档、来源、版本、状态、权限。
- `Chunk`：可检索片段，带文档来源、页码、版本、权限和向量。
- `ACL`：租户、知识空间、授权主体。
- `Citation`：最终答案引用来源。
- `AnswerTrace`：一次问答的检索、重排、模型和拒答记录。

这样可以保证：

- 权限不会被塞进松散的 metadata 后失控。
- 后续替换 LangChain、LlamaIndex、Milvus 或模型供应商时，业务模型不受影响。
- 每次回答都能审计：问了什么、召回了什么、用了什么模型、为什么拒答。

## 项目结构

```text
app/
  main.py                  FastAPI 入口，类似 Controller 层
  settings.py              应用配置
  domain/
    models.py              核心业务类型：Document、Chunk、ACL、Citation、Trace
  ports/
    contracts.py           抽象接口：存储、检索、重排、模型
  adapters/
    in_memory.py           本地内存实现，方便开发和测试
  workflows/
    indexing.py            LangGraph 文档入库工作流
    qa.py                  LangGraph 问答工作流
tests/
  test_api.py              HTTP API 测试
  test_rag_workflows.py    工作流、权限过滤、拒答和 trace 测试
```

## 快速启动

创建 conda 环境：

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

安装依赖：

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

启动服务：

```bash
uvicorn app.main:app --reload
```

打开接口文档：

- Swagger UI: http://127.0.0.1:8000/docs
- 健康检查: http://127.0.0.1:8000/health

如果当前 shell 没有激活 conda 环境，也可以这样运行：

```bash
conda run -n rag uvicorn app.main:app --reload
```

## API 示例

### 文档入库

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

### 发起问答

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

返回结果包含：

- `answer`：答案正文。
- `citations`：引用来源。
- `confidence`：当前答案置信度。
- `trace`：本次问答的流程追踪信息。

## 本地验证

```bash
pytest
ruff check .
```

或者不激活环境：

```bash
conda run -n rag pytest
conda run -n rag ruff check .
```

当前测试覆盖：

- 文档入库状态流转。
- 权限过滤：无权限 chunk 不会进入回答。
- 无授权上下文时拒答。
- 问答 trace 可追踪。
- FastAPI 入库和问答接口。

## 后续路线图

- 接入真实国产/多供应商 LLM adapter。
- 接入真实 embedding provider。
- 用 Milvus 替换内存向量检索。
- 用 OpenSearch/Elasticsearch 替换内存关键词检索。
- 用 PostgreSQL 保存文档元数据、权限、版本和问答 trace。
- 用 MinIO/S3 保存原始文件。
- 增加 PDF、Word、Excel、Markdown、HTML、网页解析。
- 增加 Redis + Celery/RQ 后台索引任务。
- 增加 Ragas 和企业标准问答集评测。
- 增加后台管理页面：文档、权限、索引状态、问答日志、反馈闭环。

## 设计原则

- 核心业务模型归项目自己所有，不被框架类型绑死。
- 权限过滤必须发生在召回阶段，不能等到生成阶段才处理。
- 有依据才回答，无依据就拒答。
- 每个答案都要能追溯到文档片段。
- 本地开发要轻，生产演进路径要清楚。
