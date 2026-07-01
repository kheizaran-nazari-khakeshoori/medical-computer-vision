"""Benchmark inference latency and throughput."""

import time
from PIL import Image

import torch

from src.inference import predict
from src.model import get_model
from src.utils import get_device


def benchmark(model=None, n_runs=20, device=None):
    device = device or get_device()
    model = model or get_model(pretrained=True).to(device)
    model.eval()
    img = Image.new("RGB", (224, 224), "gray")
    # warmup
    for _ in range(5):
        predict(img, model=model, device=device)
    start = time.perf_counter()
    for _ in range(n_runs):
        predict(img, model=model, device=device)
    elapsed = time.perf_counter() - start
    avg_ms = elapsed / n_runs * 1000
    print(f"avg latency: {avg_ms:.1f} ms over {n_runs} runs on {device}")
    print(f"throughput: {1000/avg_ms:.1f} img/s")
    return avg_ms


if __name__ == "__main__":
    benchmark()
