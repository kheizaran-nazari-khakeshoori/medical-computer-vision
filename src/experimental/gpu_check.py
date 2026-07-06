"""Checking gpu memory before training."""

import torch


def check_gpu_memory(required_mb=2000):
    if not torch.cuda.is_available():
        return True
    free = torch.cuda.mem_get_info()[0] / 1024 / 1024
    return free > required_mb


def print_gpu_info():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
