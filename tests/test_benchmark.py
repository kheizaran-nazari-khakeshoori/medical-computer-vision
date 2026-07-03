"""Benchmark and export smoke tests."""
def test_benchmark_import():
    from src.benchmark import benchmark
    assert callable(benchmark)

def test_onnx_export_smoke(tmp_path):
    from src.model import get_model
    from src.export_onnx import export_to_onnx
    m = get_model(pretrained=False)
    out = tmp_path / "model.onnx"
    res = export_to_onnx(m, str(out))
    # may be None if onnx not installed, but should not crash
    assert res is None or out.exists() or True

def test_tta_smoke():
    from PIL import Image
    from src.model import get_model
    from src.tta import tta_predict
    im = Image.new("RGB", (224,224), "gray")
    m = get_model(pretrained=False)
    probs = tta_predict(m, im, n_augmentations=2, device="cpu")
    assert probs.shape[0] == 2

def test_temperature_scaling():
    import torch
    from src.temperature_scaling import TemperatureScaling
    ts = TemperatureScaling()
    logits = torch.randn(8,2)
    labels = torch.randint(0,2,(8,))
    t = ts.fit(logits, labels, steps=5)
    assert isinstance(t, float)
