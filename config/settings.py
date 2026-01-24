"""
Settings and configuration for the YogaBharati Agentic Application.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # AI Model Configuration
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    MODEL_ID: str = os.getenv("MODEL_ID", "gpt-4.1-mini")
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "text-embedding-ada-002")
    API_ENDPOINT: str = os.getenv("API_ENDPOINT", "https://models.inference.ai.azure.com")
    API_VERSION: str = os.getenv("API_VERSION", "2024-10-21")

    # OpenSearch Configuration
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "documents")
    OPENSEARCH_USER: str = os.getenv("OPENSEARCH_USER", "")
    OPENSEARCH_PASSWORD: str = os.getenv("OPENSEARCH_PASSWORD", "")

    # YouTube Configuration (optional)
    YOUTUBE_CHANNEL_ID: str = os.getenv("YOUTUBE_CHANNEL_ID", "UCJXeL1kUCNKPiJvZaFnz6Kg")
    MAX_YOUTUBE_RESULTS: int = int(os.getenv("MAX_YOUTUBE_RESULTS", "10"))

    # Video Library Configuration
    VIDEOS_PATH: str = os.getenv("VIDEOS_PATH", "videos")

    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/agent.log")

    # RAG System Path
    RAG_SYSTEM_PATH: str = str(Path(__file__).parent.parent / "local-rag-system")

    # Agent Configuration
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "5"))
    AGENT_TIMEOUT: int = int(os.getenv("AGENT_TIMEOUT", "60"))

    def validate(self) -> None:
        """Validate required settings are configured."""
        errors = []
        
        if not self.AZURE_OPENAI_API_KEY:
            errors.append("AZURE_OPENAI_API_KEY is required")
        
        if not self.API_ENDPOINT:
            errors.append("API_ENDPOINT is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    @property
    def is_opensearch_configured(self) -> bool:
        """Check if OpenSearch is configured."""
        return bool(self.OPENSEARCH_HOST and self.OPENSEARCH_PORT)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
