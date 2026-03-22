# sagemaker_training/distillation_loss.py
"""Distillation loss with projection head and alpha annealing."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Maps student features to teacher embedding space."""

    def __init__(self, student_dim: int, teacher_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(student_dim, 512),
            nn.ReLU(),
            nn.Linear(512, teacher_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_alpha(epoch: int, total_epochs: int) -> float:
    """Linear anneal from 0.9 to 0.3 over training."""
    return 0.9 - (0.6 * epoch / (total_epochs - 1))


def distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projection_head: ProjectionHead,
    apply_distill: torch.Tensor,
) -> torch.Tensor:
    """MSE between projected student features and teacher CLS token.

    Only computed for samples where apply_distill is True.
    Returns zero if no samples have distillation applied.
    """
    mask = apply_distill.bool()
    if not mask.any():
        return torch.tensor(0.0, device=student_features.device)

    projected = projection_head(student_features[mask])
    target = teacher_features[mask].detach()
    return F.mse_loss(projected, target)


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    projection_head: ProjectionHead,
    apply_distill: torch.Tensor,
    alpha: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combined distillation + CE loss with alpha weighting."""
    ce = F.cross_entropy(logits, labels, weight=class_weights)
    distill = distillation_loss(
        student_features, teacher_features, projection_head, apply_distill
    )
    return alpha * distill + (1 - alpha) * ce
