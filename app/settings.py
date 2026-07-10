from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。

    环境变量使用 RAG_ 前缀，例如 RAG_ENVIRONMENT=prod。
    """

    app_name: str = "Enterprise RAG"
    environment: str = "local"
    default_top_k: int = 8
    llm_provider: str = "stub"
    deepseek_api_base: str = "https://api.deepseek.com"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-v4-flash"
    max_contexts: int = 3
    max_context_chars: int = 4000
    embedding_provider: str = "dashscope"
    embedding_api_base: str = (
        "https://llm-wfm5rc08u0dg1xz3.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    vector_provider: str = "milvus"
    milvus_uri: str = "http://localhost:19530"
    milvus_documents_collection: str = "rag_documents"
    milvus_chunks_collection: str = "rag_chunks"
    embedding_version: str = "text-embedding-v4-1024-v1"
    min_retrieval_score: float = 0.2
    reranker_provider: str = "dashscope"
    reranker_api_url: str = (
        "https://llm-wfm5rc08u0dg1xz3.cn-beijing.maas.aliyuncs.com/"
        "api/v1/services/rerank/text-rerank/text-rerank"
    )
    reranker_api_key: str = ""
    reranker_model: str = "qwen3-vl-rerank"
    reranker_timeout_seconds: float = 30.0
    min_rerank_score: float = 0.5
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_username: str = "root"
    mysql_password: str = ""
    mysql_database: str = "enterprise_rag"
    upload_directory: str = "data/uploads"
    upload_max_bytes: int = 20 * 1024 * 1024
    opensearch_url: str = "http://127.0.0.1:9200"
    opensearch_index: str = "rag_chunks"
    hybrid_candidate_top_k: int = 20
    hybrid_rrf_k: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")


settings = Settings()
