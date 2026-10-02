"""Step 2: split documents into chunks.

Why chunk at all? An embedding summarises a whole passage in one vector, and an LLM
has a limited, costly context window. Too big and a chunk mixes several topics so
retrieval gets fuzzy. Too small and a chunk loses the context needed to answer.

Our strategy, in order:
  1. Split Markdown by headings, so a chunk never straddles two topics. Every chunk
     remembers its heading path ("Returns and Refunds > Return window for gear").
  2. Inside a section, pack whole paragraphs into chunks of at most `max_words`.
  3. If one paragraph is too long, fall back to lines, then sentences, then raw words.
  4. When a section needs several chunks, repeat a little text at the start of each new
     chunk (`overlap_words`) so an idea cut at the boundary appears whole in one of them.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rag.config import CHUNK_OVERLAP_WORDS, CHUNK_WORDS
from rag.loaders import Document

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")

Unit = tuple[str, str]  # (text, separator to put before it when joining)


@dataclass(frozen=True)
class Chunk:
    id: str        # "<source>#<position>", stable as long as the document is unchanged
    source: str
    section: str   # heading path
    text: str
    position: int  # 0-based order within the document

    @property
    def contextual_text(self) -> str:
        """Heading path plus text. This is what we embed and keyword-index, because
        the heading often holds the words that make a chunk findable."""
        return f"{self.section}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def word_count(text: str) -> int:
    return len(text.split())


def _title_from_source(source: str) -> str:
    stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return stem.replace("-", " ").replace("_", " ").strip().title() or source


def split_sections(doc: Document) -> list[tuple[str, str]]:
    """Return (heading path, body text) for each section that has any text."""
    title = _title_from_source(doc.source)
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, str]] = []
    lines: list[str] = []
    in_code_fence = False

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append((" > ".join(h for _, h in stack) or title, body))
        lines.clear()

    for line in doc.text.split("\n"):
        if line.strip().startswith("```"):
            in_code_fence = not in_code_fence  # '#' inside code is a comment, not a heading
        match = None if in_code_fence else HEADING.match(line)
        if match:
            flush()
            level, heading = len(match.group(1)), match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            lines.append(line)
    flush()
    return sections


def _units(block: str, max_words: int, sep: str) -> list[Unit]:
    """Break a block into pieces that each fit in `max_words`, biggest natural boundary first."""
    if word_count(block) <= max_words:
        return [(block, sep)]

    lines = [ln for ln in block.split("\n") if ln.strip()]
    if len(lines) > 1:
        out: list[Unit] = []
        for i, line in enumerate(lines):
            out += _units(line, max_words, sep if i == 0 else "\n")
        return out

    sentences = [s for s in SENTENCE_SPLIT.split(block.strip()) if s]
    if len(sentences) > 1:
        out = []
        for i, sentence in enumerate(sentences):
            out += _units(sentence, max_words, sep if i == 0 else " ")
        return out

    words = block.split()  # one enormous "sentence": last resort is a hard cut
    return [(" ".join(words[i : i + max_words]), sep if i == 0 else " ") for i in range(0, len(words), max_words)]


def _render(units: list[Unit]) -> str:
    text = units[0][0]
    for piece, sep in units[1:]:
        text += sep + piece
    return text


def pack_units(units: list[Unit], max_words: int, overlap_words: int) -> list[str]:
    """Greedily fill chunks up to `max_words`, carrying trailing units over as overlap."""
    chunks: list[str] = []
    current: list[Unit] = []
    count = 0
    for unit in units:
        size = word_count(unit[0])
        if current and count + size > max_words:
            chunks.append(_render(current))
            carry: list[Unit] = []
            carried = 0
            for previous in reversed(current):
                w = word_count(previous[0])
                if carried + w > overlap_words:
                    break
                carry.insert(0, previous)
                carried += w
            while carry and carried + size > max_words:
                carried -= word_count(carry.pop(0)[0])
            current, count = carry, carried
        current.append(unit)
        count += size
    if current:
        chunks.append(_render(current))
    return chunks


def chunk_document(
    doc: Document, max_words: int = CHUNK_WORDS, overlap_words: int = CHUNK_OVERLAP_WORDS
) -> list[Chunk]:
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be at least 0 and smaller than max_words")

    chunks: list[Chunk] = []
    for section, body in split_sections(doc):
        units: list[Unit] = []
        for paragraph in PARAGRAPH_SPLIT.split(body):
            paragraph = paragraph.strip()
            if paragraph:
                units += _units(paragraph, max_words, "\n\n")
        for text in pack_units(units, max_words, overlap_words):
            position = len(chunks)
            chunks.append(Chunk(f"{doc.source}#{position}", doc.source, section, text, position))
    return chunks
