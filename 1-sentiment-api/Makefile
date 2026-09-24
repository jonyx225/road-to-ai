.PHONY: install train run test docker-build docker-run

install:
	pip install -r requirements-dev.txt

train:
	python -m app.training

run:
	uvicorn app.main:app --reload

test:
	pytest -v

docker-build:
	docker build -t sentiment-api .

docker-run:
	docker run --rm -p 8000:8000 sentiment-api
