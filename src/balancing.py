"""Class balancing for imbalanced datasets."""

from collections import Counter

from torch.utils.data import WeightedRandomSampler


def get_sampler(labels):
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
