"""Loss functions for satellite super-resolution.

Total loss = 1.00 * Charbonnier
           + 0.10 * SAM
           + 0.05 * Gradient

Each term does a different job:

Charbonnier  - how close is each pixel value?  A smooth version of L1. It is
               robust to outliers (bright roofs, water glint) so a few extreme
               pixels cannot dominate training the way they would with L2/MSE.

SAM          - is the COLOUR right?  Spectral Angle Mapper measures the angle
               between the predicted 4-band vector and the true 4-band vector at
               each pixel. It ignores brightness and only looks at the shape of
               the spectrum, which is what tells vegetation from soil from water.
               This is the term that protects the NIR band from being smeared
               into the visible bands.

Gradient     - are the EDGES sharp?  It compares horizontal and vertical
               differences between neighbouring pixels. Pixel losses alone tend
               to produce blurry output; this term pushes back against that.

All weights are configurable from configs/baseline.yaml.
"""

import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """sqrt((pred - target)^2 + eps^2) - a smooth, robust L1."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


class SAMLoss(nn.Module):
    """Spectral Angle Mapper loss: the angle between spectra, in radians.

    Dark pixels need care. A pixel whose 4-band vector is near zero has no
    direction, so its spectral angle is undefined. Dividing by its tiny norm
    gives two bad outcomes at once: identical black pixels score a confident
    90 degrees, and the gradient blows up (measured at 7.8e4 on this dataset
    before the fix). Pixels shorter than `min_norm` are therefore skipped, in
    both this loss and the matching metric in src/metrics.py.
    """

    def __init__(self, min_norm=1e-3):
        super().__init__()
        self.min_norm = min_norm

    def forward(self, pred, target):
        # pred / target: (B, C, H, W). The C values at each pixel form a vector.
        n_pred = pred.norm(dim=1)
        n_true = target.norm(dim=1)
        valid = (n_pred > self.min_norm) & (n_true > self.min_norm)
        if not valid.any():
            return pred.sum() * 0.0          # keeps the graph, contributes nothing

        dot = (pred * target).sum(dim=1)[valid]
        cos = dot / (n_pred[valid] * n_true[valid])
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)   # keep acos differentiable
        return torch.acos(cos).mean()


class GradientLoss(nn.Module):
    """L1 difference between the image gradients of prediction and target."""

    def forward(self, pred, target):
        # horizontal gradient: difference between neighbouring columns
        dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
        # vertical gradient: difference between neighbouring rows
        dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
        return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


class TotalLoss(nn.Module):
    """Weighted sum of the three terms above.

    forward() returns (total, parts) where `parts` is a dict of the individual
    unweighted values, which is handy for logging and for the notebook plots.
    """

    def __init__(self, charbonnier_weight=1.0, sam_weight=0.1, gradient_weight=0.05):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.sam = SAMLoss()
        self.gradient = GradientLoss()
        self.w_char = charbonnier_weight
        self.w_sam = sam_weight
        self.w_grad = gradient_weight

    def forward(self, pred, target):
        c = self.charbonnier(pred, target)
        s = self.sam(pred, target)
        g = self.gradient(pred, target)
        total = self.w_char * c + self.w_sam * s + self.w_grad * g
        return total, {"charbonnier": c.item(), "sam": s.item(), "gradient": g.item()}


def build_loss(cfg):
    """Create the loss from the `loss` section of the config dict."""
    l = cfg.get("loss", {})
    return TotalLoss(
        charbonnier_weight=l.get("charbonnier_weight", 1.0),
        sam_weight=l.get("sam_weight", 0.1),
        gradient_weight=l.get("gradient_weight", 0.05),
    )
