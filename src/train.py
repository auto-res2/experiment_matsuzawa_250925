import math
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CurvBlock",
    "CurvTrack",
    "softmax_entropy",
    "collect_norm_blocks",
]


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution for each sample."""
    p = F.softmax(x, dim=1)
    return -(p * torch.log(p + 1e-12)).sum(dim=1)


class CurvBlock(nn.Module):
    """Block–low-rank + diagonal inverse Fisher tracker.

    Parameters
    ----------
    shape : torch.Size
        Shape of the parameter tensor this block operates on.
    alpha : float, optional
        Exponential-moving-average coefficient for diagonal update.
    lr : float, optional
        Base learning-rate / step size.
    qbits : int, optional
        Number of quantisation bits used to store statistics (for future use).
    """

    def __init__(self, shape: torch.Size, alpha: float = 1e-4, lr: float = 3e-3, qbits: int = 8):
        super().__init__()
        n = math.prod(shape)
        # statistics live on CPU, we move to correct device on first update
        self.register_buffer("invD", torch.ones(n, dtype=torch.float32))
        self.register_buffer("u", torch.zeros(n, dtype=torch.float32))
        self.alpha = alpha
        self.lr = lr
        self.qbits = qbits
        # param handle injected later -> blk.param

    @torch.no_grad()
    def update(self, g: torch.Tensor) -> torch.Tensor:  # g is flattened gradient
        # Ensure stat tensors are on same device
        if self.invD.device != g.device:
            self.invD = self.invD.to(g.device)
            self.u = self.u.to(g.device)
        v = g.pow(2)
        denom = 1 + self.alpha * (v * self.invD)
        self.invD -= (self.alpha * (self.invD * v * self.invD)) / denom
        self.invD /= (1 - self.alpha + 1e-12)
        # dominant correlation direction (rank-1)
        self.u.mul_(0.99).add_(0.01 * (g - g.mean()))
        # natural-gradient direction in closed form (Sherman–Morrison)
        proj = self.u @ g
        denom2 = 1 + self.u @ (self.invD * self.u)
        ng = self.invD * (g - (self.u * proj) / denom2)
        return (self.lr * ng).clamp_(min=-0.1, max=0.1)


class CurvTrack(nn.Module):
    """Curvature-aware test-time adaptor wrapping a classification model."""

    def __init__(self, model: nn.Module, blocks: List[CurvBlock]):
        super().__init__()
        self.m = model
        self.blocks = blocks
        # Small helper to quickly zero relevant grads only
        self._adapt_params = [blk.param for blk in blocks]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple pass
        # Ensure we're in training mode for norm layers during adaptation
        training_mode = self.training
        if len(self._adapt_params) > 0:
            self.train()

        y = self.m(x)
        loss = softmax_entropy(y).mean()
        # Only backward if we have parameters that require gradients and loss requires grad
        if any(p.requires_grad for p in self._adapt_params) and loss.requires_grad:
            loss.backward()
            # apply NG update per block (in-place)
            for blk in self.blocks:
                if blk.param.grad is not None:
                    delta = blk.update(blk.param.grad.view(-1))
                    blk.param.add_(-delta.view_as(blk.param))
            # clear grads of adapted params only
            for p in self._adapt_params:
                if p.grad is not None:
                    p.grad = None

        # Restore original training mode
        self.train(training_mode)
        return y.detach()


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_norm_blocks(model: nn.Module, block_size: int = 16, device: str | torch.device | None = None):
    """Collect affine weights of normalisation layers and attach CurvBlocks."""
    blocks: List[CurvBlock] = []
    norm_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)
    for m in model.modules():
        if isinstance(m, norm_types) and hasattr(m, "weight") and m.weight is not None:
            p = m.weight
            p.requires_grad_(True)
            blk = CurvBlock(p.data.shape, lr=3e-3)
            blk.param = p  # type: ignore[attr-defined]
            if device is not None:
                blk.to(device)
            blocks.append(blk)
    return blocks