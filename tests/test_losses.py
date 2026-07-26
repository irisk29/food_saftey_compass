"""
Tests for the loss variants and the metric/selection logic.

Run: python -m pytest tests/ -v      (or: python tests/test_losses.py)
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.losses import AsymmetricSafetyLoss


def _fixture(n=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, generator=g) * 3
    targets = (torch.rand(n, generator=g) < 0.2).float()
    return logits, targets


def test_focal_gamma_zero_equals_pos_weight():
    """gamma=0 removes the modulating term, so focal must collapse onto pos_weight."""
    logits, targets = _fixture()
    focal = AsymmetricSafetyLoss(50.0, "focal_asymmetric", gamma=0.0)(logits, targets)
    plain = AsymmetricSafetyLoss(50.0, "pos_weight")(logits, targets)
    assert torch.allclose(focal, plain, atol=1e-6), (focal.item(), plain.item())


def test_pos_weight_matches_builtin_bce():
    """The original formulation is exactly BCEWithLogitsLoss with a positive weight."""
    logits, targets = _fixture()
    ours = AsymmetricSafetyLoss(50.0, "pos_weight")(logits, targets)

    base = nn.BCEWithLogitsLoss(reduction="none")(logits, targets)
    weights = torch.where(targets == 1, torch.full_like(targets, 50.0), torch.ones_like(targets))
    assert torch.allclose(ours, (base * weights).mean(), atol=1e-6)


def test_weight_one_is_unweighted_bce():
    """w=1 must be the plain, symmetric control for every variant."""
    logits, targets = _fixture()
    plain = nn.BCEWithLogitsLoss()(logits, targets)
    for variant in AsymmetricSafetyLoss.VARIANTS:
        got = AsymmetricSafetyLoss(1.0, variant)(logits, targets)
        assert torch.allclose(got, plain, atol=1e-6), variant


def test_error_dependent_variants_penalise_misses_more_than_hits():
    """
    The whole point of the rewrite: a confidently-correct positive should cost less
    than a confidently-wrong one under the error-dependent variants, and the same
    under pos_weight.
    """
    confident_hit = torch.tensor([6.0])    # p ~ 0.998 on a positive
    confident_miss = torch.tensor([-6.0])  # p ~ 0.002 on a positive
    positive = torch.tensor([1.0])

    for variant in ("focal_asymmetric", "fn_gated"):
        loss = AsymmetricSafetyLoss(50.0, variant)
        hit = loss(confident_hit, positive)
        miss = loss(confident_miss, positive)
        # Ratio far beyond what unweighted BCE alone would give.
        assert miss > hit * 100, (variant, hit.item(), miss.item())

    plain = AsymmetricSafetyLoss(50.0, "pos_weight")
    ratio_plain = plain(confident_miss, positive) / plain(confident_hit, positive)
    ratio_focal = (AsymmetricSafetyLoss(50.0, "focal_asymmetric")(confident_miss, positive)
                   / AsymmetricSafetyLoss(50.0, "focal_asymmetric")(confident_hit, positive))
    assert ratio_focal > ratio_plain


def test_negatives_never_upweighted():
    """The asymmetry must be one-sided: benign examples always carry weight 1."""
    logits, _ = _fixture()
    negatives = torch.zeros_like(logits)
    plain = nn.BCEWithLogitsLoss()(logits, negatives)
    for variant in AsymmetricSafetyLoss.VARIANTS:
        got = AsymmetricSafetyLoss(50.0, variant)(logits, negatives)
        assert torch.allclose(got, plain, atol=1e-6), variant


def test_gradients_are_finite():
    logits, targets = _fixture()
    for variant in AsymmetricSafetyLoss.VARIANTS:
        x = logits.clone().requires_grad_(True)
        AsymmetricSafetyLoss(50.0, variant)(x, targets).backward()
        assert torch.isfinite(x.grad).all(), variant
        assert x.grad.abs().sum() > 0, variant


def test_rejects_unknown_variant():
    try:
        AsymmetricSafetyLoss(50.0, "not_a_variant")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown variant")


def test_metrics_expose_degenerate_model():
    """
    An all-positive model must score recall 1.0 but be caught by F2, PR-AUC and the
    flag rate. This is the regression guard for the bug that motivated the change.
    """
    from src.sota_model import compute_metrics

    rng = np.random.default_rng(0)
    labels = (rng.random(1000) < 0.2).astype(int)
    all_positive_logits = np.full(1000, 20.0)  # sigmoid ~ 1.0 everywhere

    m = compute_metrics((all_positive_logits, labels))
    assert m["recall"] == 1.0, "sanity: an all-positive model does have perfect recall"
    assert m["pred_positive_rate"] == 1.0
    assert m["f2"] < 0.65, m["f2"]
    assert m["pr_auc"] < 0.35, m["pr_auc"]

    # A genuinely good model must beat it on both selection metrics.
    good_logits = np.where(labels == 1, 3.0, -3.0) + rng.normal(0, 0.5, 1000)
    good = compute_metrics((good_logits, labels))
    assert good["f2"] > m["f2"]
    assert good["pr_auc"] > m["pr_auc"]


def _logits_for(labels, n_tp, n_fp):
    """
    Decisive logits realising an exact confusion matrix: +20 (p~1) where we want a
    positive prediction, -20 (p~0) where we want a negative one.
    """
    logits = np.full(len(labels), -20.0)
    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    logits[pos_idx[:n_tp]] = 20.0
    logits[neg_idx[:n_fp]] = 20.0
    return logits


def test_f2_weights_recall_above_precision():
    """
    Two models with identical F1 but mirrored precision/recall. F2 must prefer the
    higher-recall one — that is the entire reason for choosing beta=2 over F1 here.
    """
    from src.sota_model import compute_metrics

    labels = np.zeros(1000, dtype=int)
    labels[:200] = 1

    # A: recall 0.80, precision 0.50   (TP=160, FN=40, FP=160)
    # B: recall 0.50, precision 0.80   (TP=100, FN=100, FP=25)
    a = compute_metrics((_logits_for(labels, n_tp=160, n_fp=160), labels))
    b = compute_metrics((_logits_for(labels, n_tp=100, n_fp=25), labels))

    assert abs(a["f1"] - b["f1"]) < 0.02, (a["f1"], b["f1"])   # F1 cannot separate them
    assert a["f2"] > b["f2"], (a["f2"], b["f2"])               # F2 can


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
