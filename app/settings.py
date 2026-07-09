from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。

    环境变量使用 RAG_ 前缀，例如 RAG_ENVIRONMENT=prod。
    """

    app_name: str = "Enterprise RAG"
    environment: str = "local"
    default_top_k: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_")


settings = Settings()
