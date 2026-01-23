"""
Settings and configuration for the YogaBharati Agentic Application.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # AI Model Configuration
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    MODEL_ID: str = "openai/gpt-4.1-mini"
    API_ENDPOINT: str = "https://models.inference.ai.azure.com"

    # OpenSearch Configuration
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX: str = "documents"

    # YouTube Configuration
    YOUTUBE_CHANNEL_ID: str = "UCJXeL1kUCNKPiJvZaFnz6Kg"  # YogaBharati channel
    MAX_YOUTUBE_RESULTS: int = 10

    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/agent.log"

    # RAG System Path
    RAG_SYSTEM_PATH: str = str(Path(__file__).parent.parent.parent / "local-rag-system")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
