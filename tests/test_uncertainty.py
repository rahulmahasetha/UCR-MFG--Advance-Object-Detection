import torch

from src.models.uncertainty_head import PreGraphUncertaintyHead, inverted_dropout, mc_generator


def test_reproducible_dropout():
    x = torch.ones(2, 16, 8, 8)
    a = inverted_dropout(x, 0.2, mc_generator(42, 1, 5, x.device))
    b = inverted_dropout(x, 0.2, mc_generator(42, 1, 5, x.device))
    c = inverted_dropout(x, 0.2, mc_generator(42, 1, 6, x.device))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_dual_uncertainty_maps():
    head = PreGraphUncertaintyHead(256, 10, dropout_rate=0.2, mc_passes=3)
    feats = [torch.randn(2, 256, 16, 16), torch.randn(2, 256, 8, 8), torch.randn(2, 256, 4, 4)]
    u_a, u_e, preds = head(feats, mc_generator(0, 0, 0, torch.device("cpu")))

    for ua, ue, f in zip(u_a, u_e, feats):
        assert ua.shape == ue.shape == (2, 1, *f.shape[-2:])
        assert 0 <= ua.min() and ua.max() <= 1
        assert 0 <= ue.min() and ue.max() <= 1
    assert any(ue.max() > 0 for ue in u_e), "MC dropout must produce epistemic uncertainty"
    assert set(preds) == {"log_vars", "cls_logits", "c_pres"}


def test_epistemic_disabled_is_zero():
    head = PreGraphUncertaintyHead(256, 10, mc_passes=3)
    _, u_e, _ = head([torch.randn(1, 256, 4, 4)], compute_epistemic=False)
    assert torch.all(u_e[0] == 0)
