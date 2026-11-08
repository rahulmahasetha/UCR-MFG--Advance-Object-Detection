import torch
import pytest
from src.losses.consistency_loss import ObjectCentricConsistencyLoss


def test_consistency_loss_empty_and_valid():
    loss_fn = ObjectCentricConsistencyLoss(strides=[8, 16, 32])

    # 1. Empty boxes -> Must return 0.0 without NaN and with attached autograd graph
    p1 = torch.randn(1, 10, 8, 8, requires_grad=True)
    p2 = torch.randn(1, 10, 4, 4, requires_grad=True)
    p3 = torch.randn(1, 10, 2, 2, requires_grad=True)

    empty_gt = [torch.empty((0, 4))]
    loss_empty = loss_fn([p1, p2, p3], empty_gt)
    assert not torch.isnan(loss_empty)
    assert loss_empty.item() == 0.0

    # Backprop through 0.0 attached autograd tensor
    loss_empty.backward()
    assert p1.grad is not None
    assert p1.grad.abs().sum().item() == 0.0

    # 2. Valid boxes
    p1 = torch.randn(1, 10, 8, 8, requires_grad=True)
    p2 = torch.randn(1, 10, 4, 4, requires_grad=True)
    p3 = torch.randn(1, 10, 2, 2, requires_grad=True)
    valid_gt = [torch.tensor([[10.0, 10.0, 30.0, 30.0]])]

    loss_valid = loss_fn([p1, p2, p3], valid_gt)
    assert not torch.isnan(loss_valid)
    assert loss_valid.item() >= 0.0

