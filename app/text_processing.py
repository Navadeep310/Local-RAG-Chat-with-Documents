"""File extraction and transparent, word-based chunking for workshop use."""

from dataclasses import dataclass
from io import BytesIO
import re

from docx import Document as DocxDocument
from docx.table import Table
from pypdf import PdfReader


@dataclass
class TextSegment:
    text: str
    location: str


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_pdf(data: bytes) -> list[TextSegment]:
    reader = PdfReader(BytesIO(data))
    segments = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            segments.append(TextSegment(text=text, location=f"Page {page_number}"))
    return segments


def extract_docx(data: bytes) -> list[TextSegment]:
    document = DocxDocument(BytesIO(data))
    parts = []
    for block in document.iter_inner_content():
        if isinstance(block, Table):
            parts.extend(" | ".join(cell.text for cell in row.cells) for row in block.rows)
        else:
            parts.append(block.text)
    text = normalize_text("\n".join(parts))
    return [TextSegment(text=text, location="Document")] if text else []


def extract_text(filename: str, data: bytes) -> list[TextSegment]:
    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if extension == "pdf":
            return extract_pdf(data)
        if extension == "docx":
            return extract_docx(data)
    except Exception as exc:
        raise ValueError("This file could not be read. Please check that it is a valid PDF or DOCX.") from exc
    raise ValueError("Only PDF and DOCX files are supported.")


def chunk_segments(
    segments: list[TextSegment], chunk_size: int = 220, overlap: int = 40
) -> list[TextSegment]:
    """Split text into overlapping word windows while retaining page metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[TextSegment] = []
    step = chunk_size - overlap
    for segment in segments:
        words = segment.text.split()
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size]
            if not window:
                continue
            chunks.append(TextSegment(text=" ".join(window), location=segment.location))
            if start + chunk_size >= len(words):
                break
    return chunks
