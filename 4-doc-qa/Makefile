.PHONY: install install-dev ingest ingest-offline demo eval serve serve-offline test

install:            # full app (PyTorch via sentence-transformers, anthropic, fastapi)
	pip install -r requirements.txt pytest httpx

install-dev:        # tests only
	pip install -r requirements-dev.txt

ingest:             # real semantic embeddings (downloads a ~90 MB model once)
	python -m rag.ingest

ingest-offline:     # no downloads
	python -m rag.ingest --embedder tfidf

demo: ingest-offline   # whole flow with no downloads and no API key
	python -m rag.ask "How many days do I have to return unused gear?" --llm fake

eval:
	python -m rag.evaluate --show-misses

serve:
	uvicorn rag.api:app --reload

serve-offline:
	RAG_LLM=fake uvicorn rag.api:app --reload

test:
	pytest -v
