# ============================================================
# Personal Document Finder — Pydantic Settings
# ============================================================
# Central configuration class that reads from .env file.
# Every component imports settings from here — nothing is hardcoded.
# ============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    """Application-wide configuration.

    Values are loaded from environment variables or a .env file.
    Sensible defaults are set for development; production deployments
    MUST override secrets via real env vars.
    """

    # ---- Database ----
    DATABASE_URL: str = Field(
        default="sqlite:///./documents.db",
        description="SQLAlchemy database connection string",
    )

    # ---- LLM Provider ----
    LLM_PROVIDER: Literal["openai", "ollama", "groq"] = Field(
        default="groq",
        description="Which LLM backend to use",
    )
    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API key (required when LLM_PROVIDER=openai)",
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name",
    )
    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API key (required when LLM_PROVIDER=groq)",
    )
    GROQ_MODEL: str = Field(
        default="llama-3.1-8b-instant",
        description="Groq model name",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.2",
        description="Ollama model name",
    )

    # ---- OCR ----
    OCR_ENGINE: Literal["paddleocr", "tesseract"] = Field(
        default="paddleocr",
        description="Primary OCR engine",
    )

    # ---- Vector Store ----
    VECTOR_DB: Literal["chroma"] = Field(
        default="chroma",
        description="Vector database backend",
    )
    CHROMA_PERSIST_DIR: str = Field(
        default="./chroma_data",
        description="Directory for ChromaDB persistence",
    )

    # ---- Embeddings ----
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for embeddings",
    )

    # ---- Auth / Security ----
    JWT_SECRET_KEY: str = Field(
        default="change-me-to-a-random-secret",
        description="Secret key for JWT signing — MUST be changed in production",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        description="JWT token lifetime in minutes",
    )
    SECRET_KEY: str = Field(
        default="change-me-to-another-random-secret",
        description="General-purpose application secret",
    )

    # ---- Storage ----
    STORAGE_DIR: str = Field(
        default="./storage",
        description="Root directory for uploaded documents",
    )
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        description="Maximum allowed upload size in megabytes",
    )

    # ---- Alerts ----
    EXPIRY_WARNING_DAYS: int = Field(
        default=30,
        description="Days before expiry to trigger alerts",
    )

    # ---- Server ----
    HOST: str = Field(default="0.0.0.0", description="Server bind host")
    PORT: int = Field(default=8001, description="Server bind port")
    DEBUG: bool = Field(default=True, description="Enable debug mode")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache ensures .env is read only once per process.
    Import and call this function wherever config is needed:

        from app.config.settings import get_settings
        settings = get_settings()
    """
    return Settings()
