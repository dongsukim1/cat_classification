# sagemaker_training/models.py
"""Student model factory with feature extraction support."""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

try:
    import timm
except ImportError:
    timm = None


class MobileNetV3SmallStudent(nn.Module):
    """MobileNetV3-Small with feature extraction for distillation."""

    feature_dim = 576

    def __init__(self, num_classes: int = 6):
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.extract_features(x)
        return self.classifier(feat)


class MobileNetV4ConvSStudent(nn.Module):
    """MobileNetV4-Conv-S with feature extraction for distillation.

    Note: feature_dim is 1280 (the pooled pre-logits dimension from
    mobilenetv4_conv_small.e2400_r224_in1k via timm forward_head).
    """

    feature_dim = 1280

    def __init__(self, num_classes: int = 6):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for MobileNetV4-Conv-S")
        self.base = timm.create_model(
            "mobilenetv4_conv_small.e2400_r224_in1k",
            pretrained=True,
            num_classes=num_classes,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features returns 4D (B,C,H,W); forward_head with pre_logits=True pools to 2D (B,C)
        x = self.base.forward_features(x)
        return self.base.forward_head(x, pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


class EfficientNetLite0Student(nn.Module):
    """EfficientNet-Lite0 with feature extraction for distillation.

    No SE blocks, no HardSigmoid — pure Conv+ReLU6. Designed by Google
    specifically for quantization by stripping problematic ops.
    """

    feature_dim = 1280

    def __init__(self, num_classes: int = 6):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for EfficientNet-Lite0")
        self.base = timm.create_model(
            "efficientnet_lite0",
            pretrained=True,
            num_classes=num_classes,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base.forward_features(x)
        return self.base.forward_head(x, pre_logits=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


def create_student(arch: str, num_classes: int = 6) -> nn.Module:
    """Factory function to create a student model.

    Args:
        arch: One of 'mobilenetv3_small', 'mobilenetv4_conv_s', 'efficientnet_lite0'
        num_classes: Number of output classes

    Returns:
        Student model with .extract_features() and .feature_dim
    """
    if arch == "mobilenetv3_small":
        return MobileNetV3SmallStudent(num_classes)
    elif arch == "mobilenetv4_conv_s":
        return MobileNetV4ConvSStudent(num_classes)
    elif arch == "efficientnet_lite0":
        return EfficientNetLite0Student(num_classes)
    else:
        raise ValueError(
            f"Unknown architecture: {arch}. Choose from: mobilenetv3_small, mobilenetv4_conv_s, efficientnet_lite0"
        )
