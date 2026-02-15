"""Expanding test coverage for grad-cam."""
def test_gradcam_import():
    from src.gradcam import GradCAM
    assert GradCAM is not None
def test_gradcam_plus_import():
    from src.gradcam_plus import GradCAMPlusPlus
    assert GradCAMPlusPlus is not None
