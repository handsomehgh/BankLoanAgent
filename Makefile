install:
	pip install -r requirements.txt

run:
	streamlit run app.py

clean:
	rm -rf chroma_data/ checkpoints.db

lint:
	flake8 .

