install:
	pip install -r requirements.txt

run:
	uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info

run-dev:
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

clean:
	rm -rf chroma_data/ checkpoints.db

lint:
	flake8 .

