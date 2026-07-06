"""Simplifying config management with yaml."""

from pathlib import Path

import yaml


def load_yaml(path: str):
    return yaml.safe_load(Path(path).read_text())


def save_yaml(data: dict, path: str):
    Path(path).write_text(yaml.safe_dump(data))
    return path
