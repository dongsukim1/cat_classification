# sagemaker_training/teacher.py
"""DINOv2-Large teacher model for feature distillation."""
import torch
import torch.nn as nn


class DINOv2Teacher(nn.Module):
    """Frozen DINOv2-Large that returns CLS token embeddings."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CLS token embedding (batch_size, 1024)."""
        return self.model(x)


def load_teacher(
    device: str = "cuda",
    weights_path: str | None = None,
) -> DINOv2Teacher:
    """Load frozen DINOv2-Large teacher.

    Args:
        device: Device to load model on
        weights_path: Optional local path to pre-cached weights.
                      If None, downloads from torch.hub.
    """
    if weights_path:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")

    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return DINOv2Teacher(model)
