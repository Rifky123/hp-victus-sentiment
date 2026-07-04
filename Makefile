.PHONY: install install-dev run test lint format

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

run:
	streamlit run app.py

test:
	pytest tests/ -v

lint:
	flake8 src app.py tests

format:
	black src app.py tests
