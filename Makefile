.PHONY: install run train test docker lint format
install:; pip install -r requirements.txt
run:; streamlit run app/main.py
train:; python -m src.train
test:; pytest tests -q
lint:; ruff check src app
format:; black src app && ruff check --fix src app
docker:; docker build -t radiology .
demo:; jupyter nbconvert --to notebook --execute notebooks/demo.ipynb --output /tmp/demo_out.ipynb --allow-errors
