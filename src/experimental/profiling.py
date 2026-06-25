"""Performance profiling for inference speed."""
import time
import torch
def profile_inference(model, input_tensor, n_runs=10):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(input_tensor)
            times.append(time.time() - start)
    avg = sum(times) / len(times)
    print(f"avg inference time: {avg*1000:.2f}ms over {n_runs} runs")
    return {"avg_ms": avg*1000, "times": times}
