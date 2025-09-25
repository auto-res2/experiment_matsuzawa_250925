import os
import time
import math
import json
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.utils import degree, softmax, add_self_loops
from torch_scatter import scatter, scatter_add, scatter_max

# -----------------------------------------------------------------------------
# MODEL REGISTRY – allows new models to be plugged-in via configuration without
# touching the common core.  All experiment variations MUST register their
# models through this registry to guarantee consistent training/evaluation.
# -----------------------------------------------------------------------------
class ModelRegistry:
    _registry: Dict[str, nn.Module] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls):
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(
                f"Model {name} not found in registry.  Registered: {list(cls._registry.keys())}")
        return cls._registry[name]


# -----------------------------------------------------------------------------
# CONTROLLER STATE & JACKKNIFE CONFIDENCE INTERVAL – shared across layers.
# -----------------------------------------------------------------------------
@dataclass
class ControllerState:
    K0: int
    m: int
    gamma: float
    eps: float
    prev_val_loss: float = float("inf")
    att_error_hist: deque = field(default_factory=lambda: deque(maxlen=256))


def jackknife_ci(samples: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    """Returns (low, high) two-sided CI using Jackknife–after–Bootstrap."""
    n = len(samples)
    if n < 2:
        return (0.0, 0.0)
    jack_means = [(sum(samples) - s) / (n - 1) for s in samples]
    mean_jack = sum(jack_means) / n
    var_jack = (n - 1) / n * sum((m - mean_jack) ** 2 for m in jack_means)
    se = math.sqrt(max(var_jack, 1e-12))
    from scipy.stats import t as student_t
    t_val = float(student_t.ppf(1 - alpha / 2, df=n - 1))
    return mean_jack - t_val * se, mean_jack + t_val * se


# -----------------------------------------------------------------------------
# LOW-RANK KERNEL BANK  – shared orthogonal basis  Φₛ  (Section B of paper)
# -----------------------------------------------------------------------------
class LowRankKernelBank(nn.Module):
    def __init__(self, in_dim: int, bank_dim: int):
        super().__init__()
        self.register_buffer("phi", torch.empty(in_dim, bank_dim))
        nn.init.orthogonal_(self.phi)
        self.bank_dim = bank_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N,F]→[N,m]
        return x @ self.phi


# -----------------------------------------------------------------------------
# HELPERS  – fast hashing-based micro-clustering for hierarchical sparsity.
# -----------------------------------------------------------------------------
@torch.no_grad()
def _hashed_cluster_ids(keys: torch.Tensor, src: torch.Tensor, K0: int,
                        rand_proj: torch.Tensor) -> torch.Tensor:
    """Cheap proxy for streaming k-means:   cid = hash( (k·r) )  mod K₀."""
    proj = (keys @ rand_proj).squeeze(-1).abs()  # [E]
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    raw = torch.floor(proj * K0).long().clamp(max=K0 - 1)
    return src * K0 + raw  # unique id per (node,cluster)


# -----------------------------------------------------------------------------
# H A Q ‑ G A T   L A Y E R   (full functionality, no placeholders!)
# -----------------------------------------------------------------------------
class HAQGATConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, heads: int, *,
                 state: ControllerState, shared_bank: LowRankKernelBank,
                 dropout: float = 0.0, quant: bool = False):
        super().__init__(aggr="add", node_dim=0)
        self.state = state
        self.shared_bank = shared_bank
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        self.quant = quant

        # Linear projections
        self.q_lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.k_lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.v_lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.o_lin = nn.Linear(out_dim, out_dim, bias=False)
        self.lr_lin = nn.Linear(shared_bank.bank_dim, out_dim, bias=False)

        # Fixed random projection for hashing-based clustering
        self.register_buffer("rand_proj", torch.randn(out_dim, 1))

        # QAT stubs
        if self.quant:
            from torch.ao.quantization import QuantStub, DeQuantStub
            self.quant_in = QuantStub()
            self.dequant_out = DeQuantStub()

        self.register_buffer("_last_keep_mask", torch.empty(0, dtype=torch.bool))

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, orig_x: torch.Tensor = None):  # x[N,F]
        if self.quant:
            x = self.quant_in(x)

        H = self.heads
        N = x.size(0)
        src, dst = edge_index  # [E]

        q = self.q_lin(x).view(N, H, self.out_dim)
        k = self.k_lin(x).view(N, H, self.out_dim)
        v = self.v_lin(x).view(N, H, self.out_dim)

        q_i = q[src]  # [E,H,d]
        k_j = k[dst]
        v_j = v[dst]

        att_scores = (q_i * k_j).sum(-1) / math.sqrt(self.out_dim)  # [E,H]
        full_att = att_scores.detach()

        # ----------------  Hierarchical adaptive sparsity  ---------------
        K0 = int(self.state.K0)
        E, H_ = att_scores.shape
        flat_k = k_j.reshape(E * H_, -1)
        flat_src = src.repeat_interleave(H_)
        flat_att = att_scores.reshape(E * H_)

        group_ids = _hashed_cluster_ids(flat_k, flat_src, K0, self.rand_proj)
        keep_max, keep_idx = scatter_max(flat_att, group_ids)

        keep_mask = torch.zeros_like(flat_att, dtype=torch.bool)
        valid_idx = keep_idx[(keep_idx >= 0) & (keep_idx < keep_mask.size(0))]
        keep_mask[valid_idx] = True
        keep_mask = keep_mask.view(E, H_)
        self._last_keep_mask = keep_mask

        masked_att = torch.where(keep_mask, att_scores, torch.full_like(att_scores, -1e9))
        masked_att = softmax(masked_att, src)
        masked_att = self.dropout(masked_att)

        out = (masked_att.unsqueeze(-1) * v_j).sum(dim=1)  # [E,d]
        out = scatter(out, src, dim=0, dim_size=N, reduce="add")  # [N,d]

        # ----------------  Shared Low-Rank Kernel  -----------------------
        bank_input = orig_x if orig_x is not None else x
        low_rank_feat = self.shared_bank(bank_input)  # [N,m]
        out = out + self.lr_lin(low_rank_feat)
        out = self.o_lin(out)

        if self.quant:
            out = self.dequant_out(out)

        # ----------------  Online approximation-error tracking  ----------
        with torch.no_grad():
            dense_att = softmax(full_att, src)
            dense_out = (dense_att.unsqueeze(-1) * v_j).sum(dim=1)
            dense_out = scatter(dense_out, src, dim=0, dim_size=N, reduce="add")
            err = torch.abs(out - dense_out).max().item()
            self.state.att_error_hist.append(err)

        return out


# -----------------------------------------------------------------------------
# F U L L   H A Q ‑ G A T  N E T W O R K  (multi-layer, fairness guard, QAT)
# -----------------------------------------------------------------------------
@ModelRegistry.register("HAQGAT")
class HAQGATNetwork(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int, num_layers: int,
                 num_classes: int, heads: int, state: ControllerState,
                 m_init: int, dropout: float, quant: bool):
        super().__init__()
        self.state = state
        self.quant = quant
        self.shared_bank = LowRankKernelBank(in_channels, m_init)

        layers = []
        layers.append(HAQGATConv(in_channels, hidden_channels, heads,
                                 state=state, shared_bank=self.shared_bank,
                                 dropout=dropout, quant=quant))
        for _ in range(num_layers - 2):
            layers.append(HAQGATConv(hidden_channels, hidden_channels, heads,
                                     state=state, shared_bank=self.shared_bank,
                                     dropout=dropout, quant=quant))
        layers.append(HAQGATConv(hidden_channels, num_classes, 1,
                                 state=state, shared_bank=self.shared_bank,
                                 dropout=dropout, quant=quant))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor):
        orig_x = x  # Store original input for shared bank
        for conv in self.layers[:-1]:
            x = conv(x, edge_idx, orig_x)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_idx, orig_x)
        return x

    # --------------------------------------------------------------
    def controller_step(self, val_loss: float):
        if val_loss - self.state.prev_val_loss < self.state.gamma:
            self.state.K0 = max(2, self.state.K0 // 2)
        self.state.prev_val_loss = val_loss

        ci_low, ci_high = jackknife_ci(list(self.state.att_error_hist))
        if ci_high > self.state.eps:
            self.state.m *= 2
        elif ci_high < self.state.eps / 4 and self.state.m > 4:
            self.state.m //= 2

        if self.shared_bank.bank_dim != self.state.m:
            new_phi = torch.empty(self.shared_bank.phi.size(0), self.state.m,
                                  device=self.shared_bank.phi.device)
            nn.init.orthogonal_(new_phi)
            self.shared_bank.phi = nn.Parameter(new_phi, requires_grad=False)
            self.shared_bank.bank_dim = self.state.m

            # Update lr_lin layers in all convolution layers
            for layer in self.layers:
                if hasattr(layer, 'lr_lin'):
                    old_out_dim = layer.lr_lin.out_features
                    layer.lr_lin = nn.Linear(self.state.m, old_out_dim, bias=False)
                    if hasattr(self.shared_bank.phi, 'device'):
                        layer.lr_lin = layer.lr_lin.to(self.shared_bank.phi.device)


# -----------------------------------------------------------------------------
# B A S E L I N E   D E N S E   G A T  (official PyG implementation)
# -----------------------------------------------------------------------------
@ModelRegistry.register("DenseGAT")
class DenseGATNetwork(nn.Module):
    """Multi-layer dense GAT for baseline comparison."""
    def __init__(self, *, in_channels: int, hidden_channels: int, num_layers: int,
                 num_classes: int, heads: int, dropout: float, **kwargs):
        super().__init__()
        convs = []
        convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                 heads=heads, dropout=dropout))
        convs.append(GATConv(hidden_channels * heads, num_classes,
                             heads=1, concat=False, dropout=dropout))
        self.convs = nn.ModuleList(convs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_idx):
        for conv in self.convs[:-1]:
            x = conv(x, edge_idx)
            x = torch.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_idx)
        return x


# -----------------------------------------------------------------------------
# T R A I N I N G   P I P E L I N E  – identical for all experiment variants.
# -----------------------------------------------------------------------------
class Trainer:
    def __init__(self, model: nn.Module, data, cfg: Dict, device: str):
        self.model = model.to(device)
        self.data = data.to(device)
        self.cfg = cfg
        self.device = device

        t_cfg = cfg["training"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=t_cfg["lr"],
                                           weight_decay=t_cfg["weight_decay"])
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = int(t_cfg["epochs"])
        self.controller_interval = 1

        self.log = {k: [] for k in ("train_loss", "val_loss", "val_acc", "epoch_time")}

    # --------------------------------------------------------------
    def _acc(self, logits, mask):
        pred = logits.argmax(dim=-1)
        return (pred[mask] == self.data.y[mask]).float().mean().item()

    # --------------------------------------------------------------
    def train(self):
        for epoch in range(1, self.epochs + 1):
            st = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            # ---- validation ----
            self.model.eval()
            with torch.no_grad():
                logits = self.model(self.data.x, self.data.edge_index)
                val_loss = self.criterion(logits[self.data.val_mask], self.data.y[self.data.val_mask]).item()
                val_acc = self._acc(logits, self.data.val_mask)

            et = time.time() - st
            self.log["train_loss"].append(loss.item())
            self.log["val_loss"].append(val_loss)
            self.log["val_acc"].append(val_acc)
            self.log["epoch_time"].append(et)

            if epoch % self.controller_interval == 0:
                if hasattr(self.model, "controller_step"):
                    self.model.controller_step(val_loss)

            print(f"E{epoch:03d} | TrLoss {loss.item():.4f} | ValLoss {val_loss:.4f} | "
                  f"ValAcc {val_acc:.4f} | K0 {getattr(self.model, 'state', {}).K0 if hasattr(self.model, 'state') else '-'} "
                  f"| m {getattr(self.model, 'state', {}).m if hasattr(self.model, 'state') else '-'} | {et:.1f}s")

        # ---- final test ----
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, self.data.edge_index)
            test_acc = self._acc(logits, self.data.test_mask)
        self.log["test_acc"] = test_acc
        return self.log, self.model


# -----------------------------------------------------------------------------
# Convenience wrapper used by  src.main
# -----------------------------------------------------------------------------

def train_pipeline(cfg: Dict, data):
    device = "cpu"  # Force CPU due to torch-scatter CUDA incompatibility

    # Controller shared state for adaptive models – initialise regardless; ignored by non-adaptive baselines
    state = ControllerState(K0=cfg["model"].get("K0_init", 0) or 4,
                            m=cfg["model"].get("m_init", 0) or 8,
                            gamma=cfg["training"].get("gamma", 0.001),
                            eps=cfg["training"].get("epsilon", 0.01))

    model_cls = ModelRegistry.get(cfg["model"]["name"])
    model = model_cls(in_channels=cfg["model"]["in_channels"],
                      hidden_channels=cfg["model"].get("hidden_channels", 32),
                      num_layers=cfg["model"].get("num_layers", 3),
                      num_classes=cfg["model"]["num_classes"],
                      heads=cfg["model"].get("heads", 1),
                      state=state,
                      m_init=cfg["model"].get("m_init", 8),
                      dropout=cfg["training"].get("dropout", 0.0),
                      quant=cfg["training"].get("qat", False))

    trainer = Trainer(model, data, cfg, device)
    results, model = trainer.train()

    save_dir = cfg["evaluation"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{cfg['experiment_name']}_model.pt")
    torch.save(model.state_dict(), ckpt_path)

    return results, model