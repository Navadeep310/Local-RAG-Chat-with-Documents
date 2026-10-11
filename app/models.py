from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Chunk:
    id: str
    document_id: str
    document_name: str
    text: str
    embedding: list[float]
    location: str


@dataclass
class Document:
    id: str
    name: str
    file_type: str
    size_bytes: int
    chunks: list[Chunk]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "file_type": self.file_type,
            "size_bytes": self.size_bytes,
            "chunk_count": len(self.chunks),
            "created_at": self.created_at,
        }


def new_document_id() -> str:
    return str(uuid4())
