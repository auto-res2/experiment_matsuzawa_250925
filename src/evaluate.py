import json
import os
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

__all__ = ["evaluate"]


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    n = 0
    tic = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.numel()
    dur = time.perf_counter() - tic
    acc = correct / n
    res = {"accuracy": acc, "num_samples": n, "eval_seconds": dur}
    print("Evaluation results:", json.dumps(res, indent=2))
    return res
