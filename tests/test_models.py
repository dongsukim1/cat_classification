# tests/test_models.py
import torch
import pytest


def test_mobilenetv3_small_output_shape():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv3_small", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 6)


def test_mobilenetv3_small_features():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv3_small", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_features(x)
    assert features.shape == (2, 576)


def test_mobilenetv4_conv_s_output_shape():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv4_conv_s", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 6)


def test_mobilenetv4_conv_s_features():
    from sagemaker_training.models import create_student

    model = create_student("mobilenetv4_conv_s", num_classes=6)
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_features(x)
    assert features.shape == (2, 1280)


def test_feature_dim_attribute():
    from sagemaker_training.models import create_student

    mv3 = create_student("mobilenetv3_small", num_classes=6)
    assert mv3.feature_dim == 576

    mv4 = create_student("mobilenetv4_conv_s", num_classes=6)
    assert mv4.feature_dim == 1280


def test_invalid_arch_raises():
    from sagemaker_training.models import create_student

    with pytest.raises(ValueError):
        create_student("invalid_arch", num_classes=6)
