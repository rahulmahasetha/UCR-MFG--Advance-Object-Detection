import torch
import pytest
from src.models.uncertainty_head import (
    PreGraphUncertaintyHead,
    apply_reproducible_dropout,
    get_batch_mc_generator
)


def test_reproducible_dropout():
    x = torch.ones((2, 16, 8, 8))
    device = x.device
    gen1 = get_batch_mc_generator(base_seed=42, epoch=1, batch_idx=5, t=0, device=device)
    out1 = apply_reproducible_dropout(x, p=0.2, generator=gen1)

    gen2 = get_batch_mc_generator(base_seed=42, epoch=1, batch_idx=5, t=0, device=device)
    out2 = apply_reproducible_dropout(x, p=0.2, generator=gen2)

    # Identical seed hash produces bit-exact masks
    assert torch.equal(out1, out2)

    gen3 = get_batch_mc_generator(base_seed=42, epoch=1, batch_idx=6, t=0, device=device)
    out3 = apply_reproducible_dropout(x, p=0.2, generator=gen3)
    # Different batch produces different mask
    assert not torch.equal(out1, out3)


def test_uncertainty_head_outputs():
    head = PreGraphUncertaintyHead(in_channels=256, num_classes=10, dropout_rate=0.2, mc_passes=3)
    f1 = torch.randn(2, 256, 16, 16)
    f2 = torch.randn(2, 256, 8, 8)
    f3 = torch.randn(2, 256, 4, 4)

    ua_list, ue_list, preds = head((f1, f2, f3))
    assert len(ua_list) == 3
    assert len(ue_list) == 3

    for lvl, (ua, ue) in enumerate(zip(ua_list, ue_list)):
        # Bounded in [0, 1]
        assert ua.min() >= 0.0 and ua.max() <= 1.0
        assert ue.min() >= 0.0 and ue.max() <= 1.0
        assert ua.shape == (2, 1, f1.shape[2] // (2**lvl), f1.shape[3] // (2**lvl))
        assert ue.shape == (2, 1, f1.shape[2] // (2**lvl), f1.shape[3] // (2**lvl))

    assert "log_vars" in preds
    assert "cls_logits" in preds
    assert "c_pres" in preds

