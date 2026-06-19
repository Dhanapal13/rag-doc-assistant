from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "RAG Document Assistant"
    environment: str = "development"  # development | staging | production
    chroma_db_path: str = "/app/chroma_db"
    ollama_base_url: str = "http://localhost:11434"
    ollama_request_timeout: int = 300
    default_llm_model: str = "llama3.2:3b"
    gemma_model: str = "gemma2:2b"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    max_question_length: int = 1000
    cors_allowed_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"

settings = Settings()