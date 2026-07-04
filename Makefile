.PHONY: setup data reference feast train serve producer consumer retrain test lint docker docker-down clean

setup:
	python3.11 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

data:
	python -m src.data.preprocess

reference:
	python scripts/generate_reference_dataset.py

validate:
	python -m src.validation.expectations

feast:
	cd src/features/feature_repo && feast apply && cd ../../..
	python -m src.features.materializer

train:
	python -m src.training.train

serve:
	uvicorn src.serving.main:app --reload --host 0.0.0.0 --port 8000

producer:
	python -m src.producer.flow_producer

consumer:
	python -m src.consumer.flow_consumer

retrain:
	python -m src.orchestration.flows.retraining_flow

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

test:
	pytest tests/ -v

test-ci:
	pytest tests/ -m "not requires_data" -v

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

benchmark:
	locust -f benchmark/locustfile.py --headless -u 50 -r 10 --run-time 60s --host http://localhost:8000

docker:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f predictions.db
