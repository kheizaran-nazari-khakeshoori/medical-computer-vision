.PHONY: install run train test docker
install:; pip install -r requirements.txt
run:; streamlit run app/main.py
train:; python -m src.train
test:; pytest tests -q
docker:; docker build -t radiology .
