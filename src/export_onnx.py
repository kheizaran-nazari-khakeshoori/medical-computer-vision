"""ONNX export for model deployment."""

import torch


def export_to_onnx(model, output_path: str = "models/model.onnx", input_size: tuple = (1, 3, 224, 224)):
    model.eval()
    dummy_input = torch.randn(*input_size)
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
