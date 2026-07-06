"""Connecting frontend to backend api."""

import requests


def call_predict_api(image_path: str, api_url="http://localhost:8000/predict"):
    with open(image_path, "rb") as f:
        r = requests.post(api_url, files={"file": f})
        return r.json()
