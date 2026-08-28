.PHONY: install install-dev test lint download split vocab features train evaluate pipeline api gradio docker-build docker-run clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

download:
	python scripts/download_data.py

split:
	python scripts/split_dataset.py

vocab:
	python scripts/build_vocab.py

features:
	python scripts/extract_features.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

pipeline:
	bash scripts/run_pipeline.sh

api:
	uvicorn captioner.serving.api:app --reload --host 0.0.0.0 --port 8000

gradio:
	python -m captioner.serving.app_gradio

docker-build:
	docker build -t flickr8k-caption-gen .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts flickr8k-caption-gen

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
