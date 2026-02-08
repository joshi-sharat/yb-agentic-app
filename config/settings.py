"""
Settings and configuration for the YogaBharati Agentic Application.
Updated to use Anthropic Claude instead of Azure OpenAI.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # RAG Service URLs
    RAG_SERVICE_URL: str = "http://localhost:8080"
    RAG_UI_URL: str = "http://localhost:8501"
    
    # Logging
    LOG_FILE_PATH: str = "logs/app.log"  # File path for the application log file

    # For Ollama 
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://192.168.4.115:11434")
    
    # Embedding Configuration
    # Using local sentence-transformers model (no API key needed)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # OpenSearch Configuration
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "documents")
    OPENSEARCH_USER: str = os.getenv("OPENSEARCH_USER", "")
    OPENSEARCH_PASSWORD: str = os.getenv("OPENSEARCH_PASSWORD", "")
    
    # YouTube Configuration (optional)
    YOUTUBE_CHANNEL_ID: str = os.getenv("YOUTUBE_CHANNEL_ID", "UCJXe1kUCNKPiJvZaFnz6Kg")
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
    
    # LLM Provider Selection
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # Options: "anthropic", "azure", "openai"
    
    def validate(self) -> None:
        """Validate required settings are configured."""
        errors = []
        
        # Check based on selected LLM provider
        # if self.LLM_PROVIDER == "anthropic":
        #     if not self.ANTHROPIC_API_KEY:
        #         errors.append("ANTHROPIC_API_KEY is required when using Anthropic")
        # elif self.LLM_PROVIDER in ["azure", "openai"]:
        #     if not self.AZURE_OPENAI_API_KEY:
        #         errors.append("AZURE_OPENAI_API_KEY is required when using Azure/OpenAI")
        #     if not self.API_ENDPOINT:
        #         errors.append("API_ENDPOINT is required")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    @property
    def is_opensearch_configured(self) -> bool:
        """Check if OpenSearch is configured."""
        return bool(self.OPENSEARCH_HOST and self.OPENSEARCH_PORT)
    
    @property
    def is_ollama_configured(self) -> bool:
        """Check if Anthropic API is configured."""
        return bool(self.OLLAMA_HOST and self.OLLAMA_MODEL)


    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create singleton settings instance
settings = Settings()