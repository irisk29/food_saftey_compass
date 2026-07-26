"""
Loss variants for hazard detection, built so the three can be compared directly.

The original `AsymmetricSafetyLoss` was documented as penalising *false negatives*,
but its mask keyed on the label (`weight_mask[targets == 1.0] = w`), which upweights
every positive example whether the model got it right or not. That is exactly
`nn.BCEWithLogitsLoss(pos_weight=w)` — standard cost-sensitive class weighting, not
an error-dependent penalty.

Rather than quietly rename it, we keep that formulation as an explicit baseline
(`pos_weight`) and add two variants whose weight genuinely depends on the model's
current error, so "penalise false negatives" becomes a true statement:

  pos_weight       w(y=1) = w_fn                                (label-dependent)
  focal_asymmetric w(y=1) = 1 + (w_fn - 1) * (1 - p)^gamma      (error-dependent, smooth)
  fn_gated         w(y=1) = w_fn if p < tau else 1.0            (error-dependent, hard)

where p = sigmoid(logit) is the model's current hazard probability. Negatives always
carry weight 1.0 in all three — the asymmetry is the point.

Why the two variants differ, and why we report both:
  - `focal_asymmetric` never fully switches the penalty off, so easy positives still
    contribute a little. gamma controls how sharply weight concentrates on hard
    positives (gamma=0 collapses exactly back to `pos_weight`, which is a useful
    sanity check).
  - `fn_gated` is the literal reading: pay the penalty only on examples that are
    *actually* false negatives at the deployment threshold tau. Cleaner to explain,
    but the hard gate makes the effective batch weight jump around during training.

Setting gamma=0 in `focal_asymmetric` must reproduce `pos_weight` numerically; this
is asserted in `tests/test_losses.py`.
"""

import torch
import torch.nn as nn


class AsymmetricSafetyLoss(nn.Module):
    """
    Binary cross-entropy with an asymmetric penalty on the hazard class.

    Args:
        weight_fn: penalty multiplier applied to positive (hazard) examples.
        variant:   'pos_weight' | 'focal_asymmetric' | 'fn_gated'
        gamma:     focusing exponent, only used by 'focal_asymmetric'.
        tau:       probability gate, only used by 'fn_gated'. Should match the
                   decision threshold the model is actually deployed at.
    """

    VARIANTS = ("pos_weight", "focal_asymmetric", "fn_gated")

    def __init__(self, weight_fn=50.0, variant="pos_weight", gamma=2.0, tau=0.20):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown loss variant {variant!r}; expected one of {self.VARIANTS}")
        self.weight_fn = float(weight_fn)
        self.variant = variant
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _positive_weights(self, logits, targets):
        """Per-example multiplier for the positive class. Negatives handled by caller."""
        if self.variant == "pos_weight":
            # Label-dependent: every hazard example weighted identically.
            return torch.full_like(targets, self.weight_fn)

        probs = torch.sigmoid(logits)

        if self.variant == "focal_asymmetric":
            # Error-dependent and smooth. A positive the model already scores near 1.0
            # contributes weight ~1; one scored near 0 (a bad miss) contributes ~w_fn.
            # Gradient intentionally flows through the modulating term, as in the
            # focal-loss literature (Lin et al., 2017).
            return 1.0 + (self.weight_fn - 1.0) * (1.0 - probs).pow(self.gamma)

        # fn_gated: pay the penalty only on examples that are false negatives at tau.
        # The gate is a hard, non-differentiable mask, so it is detached explicitly —
        # gradient still flows through the underlying BCE term.
        is_fn = (probs < self.tau).detach().to(targets.dtype)
        return 1.0 + (self.weight_fn - 1.0) * is_fn

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        base_loss = self.bce(logits, targets)

        pos_w = self._positive_weights(logits, targets)
        # Negatives always weight 1.0; only the hazard class carries the penalty.
        weights = torch.where(targets == 1.0, pos_w, torch.ones_like(targets))

        return (base_loss * weights).mean()

    def extra_repr(self):
        bits = [f"variant={self.variant}", f"weight_fn={self.weight_fn}"]
        if self.variant == "focal_asymmetric":
            bits.append(f"gamma={self.gamma}")
        if self.variant == "fn_gated":
            bits.append(f"tau={self.tau}")
        return ", ".join(bits)
