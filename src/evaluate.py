"""src/evaluate.py
Unified evaluation – now fixed to *accuracy* for multi-class node-classification
(which covers all OGB datasets) and AUROC only for binary problems.
"""
from __future__ import annotations
import torch, torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.data import Data


def _np(x: torch.Tensor):
    return x.detach().cpu().numpy()


class Evaluator:
    def __init__(self, data: Data):
        self.data = data

    @staticmethod
    def _batch_iter(idx: torch.Tensor, bs: int):
        for i in range(0, idx.numel(), bs):
            yield idx[i : i + bs]

    def eval(self, model: torch.nn.Module, split: str = "test", batch_size: int = 8192):
        mask = getattr(self.data, f"{split}_mask")
        idx = mask.nonzero(as_tuple=False).view(-1)
        preds, labels = [], []
        for b in self._batch_iter(idx, batch_size):
            out = model(self.data.x, self.data.edge_index)[b]
            preds.append(out)
            labels.append(self.data.y[b])
        logits = torch.cat(preds, dim=0)
        y_true = torch.cat(labels, dim=0)
        num_classes = logits.shape[1]
        if num_classes > 2:
            y_hat = logits.argmax(dim=1)
            return accuracy_score(_np(y_true), _np(y_hat))
        else:
            y_prob = F.softmax(logits, dim=1)[:, 1]
            return roc_auc_score(_np(y_true), _np(y_prob))
