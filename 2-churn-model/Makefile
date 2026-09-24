.PHONY: install data explore train predict test all

install:
	pip install -r requirements-dev.txt

data:
	python -m churn.data

explore:
	python -m churn.explore

train:
	python -m churn.train

predict:
	python -m churn.predict --input data/churn.csv --top 10

test:
	pytest -v

all: data explore train
