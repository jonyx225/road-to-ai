"""Step 1: read documents from disk.

Supported: Markdown (.md), plain text (.txt) and PDF (.pdf, needs `pip install pypdf`).
Each document's `source` is its path relative to the docs folder, and that string is
what we show as the citation later.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass(frozen=True)
class Document:
    source: str   # e.g. "returns-and-refunds.md"
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(f"Reading {path.name} needs pypdf: pip install pypdf") from exc
    pages = [(page.extract_text() or "").strip() for page in PdfReader(str(path)).pages]
    return "\n\n".join(p for p in pages if p)


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 text") from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_documents(path: Path | str) -> list[Document]:
    """Load one file, or every supported file under a folder (sorted, recursive)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if path.is_file():
        root, files = path.parent, [path]
    else:
        root = path
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file()
            and p.suffix.lower() in SUPPORTED_SUFFIXES
            and not any(part.startswith(".") for part in p.relative_to(path).parts)
        )

    documents = []
    for file in files:
        text = _read(file)
        if not text.strip():
            logger.warning("Skipping %s: no text found (a scanned PDF needs OCR first)", file)
            continue
        documents.append(Document(source=file.relative_to(root).as_posix(), text=text))

    if not documents:
        raise ValueError(f"No readable documents (.md, .txt, .pdf) found in {path}")
    return documents
