"""Configuration for the Smart Customer Support Chatbot System."""
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    LLM_PROVIDER: str       = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str     = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str          = os.getenv("LLM_MODEL", "gpt-4o-mini")
    STRONG_MODEL: str       = os.getenv("STRONG_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    CHROMA_PATH: str        = os.getenv("CHROMA_PATH", "./chroma_db")
    DOCS_PATH: str          = os.getenv("DOCS_PATH", "./sample_docs")

    CHUNK_SIZE: int         = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int      = int(os.getenv("CHUNK_OVERLAP", "64"))
    TOP_K: int              = int(os.getenv("TOP_K", "4"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.62"))

    MAX_HISTORY_TURNS: int  = int(os.getenv("MAX_HISTORY_TURNS", "8"))
    TEMPERATURE: float      = float(os.getenv("TEMPERATURE", "0.1"))

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

    # API key for client authentication (comma-separated list)
    API_KEYS: str           = os.getenv("API_KEYS", "demo-key-12345")

    APP_PORT: int           = int(os.getenv("APP_PORT", "8001"))
    APP_HOST: str           = os.getenv("APP_HOST", "0.0.0.0")
    DEBUG: bool             = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
