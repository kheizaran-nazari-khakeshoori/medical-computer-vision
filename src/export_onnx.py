"""ONNX export for model deployment."""

import os
from pathlib import Path

import torch

from src.utils import ensure_dir


def export_to_onnx(model, output_path: str = "models/model.onnx", input_size: tuple = (1, 3, 224, 224)):
    model.eval()
    ensure_dir(Path(output_path).parent)
    dummy_input = torch.randn(*input_size)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        print(f"exported onnx model to {output_path}")
        return output_path
    except Exception as e:
        print(f"onnx export failed: {e} (install onnx via pip install onnx)")
        return None


if __name__ == "__main__":
    from src.model import get_model

    m = get_model(pretrained=False)
    export_to_onnx(m)
