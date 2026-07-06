"""Integrating prometheus metrics."""

try:
    from prometheus_client import Counter, Histogram

    PREDICT_COUNTER = Counter("predictions_total", "Total predictions")
    LATENCY_HIST = Histogram("inference_latency_seconds", "Inference latency")
except Exception:

    class Dummy:
        def inc(self, *a, **k):
            pass

        def observe(self, *a, **k):
            pass

        def labels(self, *a, **k):
            return self

        def time(self):
            return self

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    PREDICT_COUNTER = Dummy()
    LATENCY_HIST = Dummy()
