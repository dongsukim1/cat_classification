# tests/test_distillation_loss.py
import torch
import pytest


def test_projection_head_shape():
    from sagemaker_training.distillation_loss import ProjectionHead

    head = ProjectionHead(student_dim=576, teacher_dim=1024)
    x = torch.randn(4, 576)
    out = head(x)
    assert out.shape == (4, 1024)


def test_alpha_annealing():
    from sagemaker_training.distillation_loss import compute_alpha

    # Epoch 0 of 20: should be 0.9
    assert abs(compute_alpha(0, 20) - 0.9) < 1e-6
    # Last epoch: should be 0.3
    assert abs(compute_alpha(19, 20) - 0.3) < 1e-6
    # Midpoint: should be 0.6
    assert abs(compute_alpha(9, 20) - 0.6) < 0.05


def test_distillation_loss_with_mask():
    from sagemaker_training.distillation_loss import distillation_loss, ProjectionHead

    student_features = torch.randn(4, 576)
    teacher_features = torch.randn(4, 1024)
    apply_distill = torch.tensor([True, True, False, False])

    head = ProjectionHead(576, 1024)
    loss = distillation_loss(student_features, teacher_features, head, apply_distill)

    assert loss.shape == ()  # scalar
    assert loss.item() >= 0


def test_distillation_loss_all_masked_returns_zero():
    from sagemaker_training.distillation_loss import distillation_loss, ProjectionHead

    head = ProjectionHead(576, 1024)
    student_features = torch.randn(4, 576)
    teacher_features = torch.randn(4, 1024)
    apply_distill = torch.tensor([False, False, False, False])

    loss = distillation_loss(student_features, teacher_features, head, apply_distill)
    assert loss.item() == 0.0


def test_combined_loss():
    from sagemaker_training.distillation_loss import combined_loss, ProjectionHead

    head = ProjectionHead(576, 1024)
    student_features = torch.randn(4, 576)
    teacher_features = torch.randn(4, 1024)
    logits = torch.randn(4, 6)
    labels = torch.tensor([0, 1, 2, 3])
    apply_distill = torch.tensor([True, True, True, False])

    loss = combined_loss(
        logits=logits,
        labels=labels,
        student_features=student_features,
        teacher_features=teacher_features,
        projection_head=head,
        apply_distill=apply_distill,
        alpha=0.7,
        class_weights=None,
    )
    assert loss.shape == ()
    assert loss.item() > 0
