"""Examining failure cases with error analysis."""

from collections import Counter


def analyze_failures(y_true, y_pred):
    failures = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    return Counter(failures)


def failure_rate(y_true, y_pred):
    return sum(t != p for t, p in zip(y_true, y_pred)) / len(y_true)
