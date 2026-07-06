"""Setting up environment variables for api keys."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    pass


def get_env(key: str, default=None):
    return os.getenv(key, default)


MODEL_PATH = get_env("MODEL_PATH", "models/resnet50_medical.pth")
API_KEY = get_env("API_KEY", "")
