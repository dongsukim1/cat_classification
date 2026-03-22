# tests/test_teacher.py
import torch
import pytest


@pytest.mark.slow
def test_teacher_output_shape():
    from sagemaker_training.teacher import load_teacher

    teacher = load_teacher(device="cpu")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = teacher(x)
    assert features.shape == (2, 1024)


@pytest.mark.slow
def test_teacher_is_frozen():
    from sagemaker_training.teacher import load_teacher

    teacher = load_teacher(device="cpu")
    for param in teacher.parameters():
        assert not param.requires_grad


@pytest.mark.slow
def test_teacher_deterministic():
    from sagemaker_training.teacher import load_teacher

    teacher = load_teacher(device="cpu")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out1 = teacher(x)
        out2 = teacher(x)
    assert torch.allclose(out1, out2)
