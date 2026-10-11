from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "220"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))


settings = Settings()
