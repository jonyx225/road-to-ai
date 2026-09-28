"""Shared test helpers. Everything here is offline: tf-idf embeddings and a fake LLM."""
from functools import lru_cache
from types import SimpleNamespace

from rag import config
from rag.embeddings import create_embedder
from rag.loaders import Document, load_documents
from rag.store import Index

DOCS = config.BASE_DIR / "data" / "docs"


@lru_cache(maxsize=1)
def sample_index() -> Index:
    """Index of the bundled sample documents (built once, shared by all tests)."""
    return Index.build(load_documents(DOCS), create_embedder("tfidf"), chunk_words=160, overlap_words=30)


def make_doc(source: str, text: str) -> Document:
    return Document(source=source, text=text)


class ScriptedClient:
    """Stands in for anthropic.Anthropic(): always replies with `reply` and records the calls."""

    def __init__(self, reply: str, input_tokens: int = 100, output_tokens: int = 10):
        self.reply, self.calls = reply, []
        self.input_tokens, self.output_tokens = input_tokens, output_tokens
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.reply)],
            usage=SimpleNamespace(input_tokens=self.input_tokens, output_tokens=self.output_tokens),
        )


class FailingClient:
    def __init__(self):
        self.messages = self

    def create(self, **kwargs):
        raise ConnectionError("network down")
