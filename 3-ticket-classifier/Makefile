.PHONY: install install-core data baseline finetune llm-small compare test

install:            # full project (torch, transformers, anthropic)
	pip install -r requirements.txt pytest

install-core:       # data + baseline + tests only
	pip install -r requirements-core.txt

data:
	python -m tickets.data

baseline:
	python -m tickets.compare --methods baseline

finetune:
	python -m tickets.finetune

llm-small:          # cheap first LLM run on 50 tickets
	python -m tickets.compare --methods llm --limit 50

compare:
	python -m tickets.compare

test:
	pytest -v
