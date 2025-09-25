"""src/train.py
Specialised training logic for the concrete TADP-HABLS experiments of the
paper.  All generic hooks have been replaced by *working* implementations
for the OGBN-Products / Papers100M / Friendster datasets and the GAT /
GAT-v2 / Graph-Transformer / GeniePath encoders built on PyTorch-Geometric.
"""
from __future__ import annotations

import json, time, math, random
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.nn import (
    GATConv,      # GAT
    GATv2Conv,    # GAT-v2
    TransformerConv,  # Graph Transformer (approx.)
    GeniePathConv,     # GeniePath layer
)
from tqdm import tqdm

from .preprocess import load_dataset
from .evaluate import Evaluator
from .utils.energy import EnergyMeter
from .utils.metrics import batch_jaccard, neighbour_variance
from .utils.seed import seed_everything
from .utils.figures import plot_training_curves, plot_metric_bars

################################################################################
#                         Models for all experimental runs                     #
################################################################################

class GATNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads,
                                      dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, num_classes, heads=1,
                                  concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATv2Net(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 4, heads: int = 8, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                                        dropout=dropout))
        self.convs.append(GATv2Conv(hidden_dim * heads, num_classes, heads=1,
                                    concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GraphTransformerNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 8, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim * heads, hidden_dim,
                                              heads=heads, dropout=dropout))
        self.convs.append(TransformerConv(hidden_dim * heads, num_classes,
                                          heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GeniePathNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GeniePathConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GeniePathConv(hidden_dim, hidden_dim))
        self.convs.append(GeniePathConv(hidden_dim, num_classes))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

################################################################################
#                                 Sampler Core (unchanged)                     #
################################################################################
from .sampler import TADPSampler       # moved to its own file for clarity

################################################################################
#                        Early-Stopping helper (unchanged)                    #
################################################################################
class EarlyStopper:
    def __init__(self, patience: int = 50):
        self.best = -float('inf')
        self.best_epoch = 0
        self.patience = patience

    def step(self, metric: float, epoch: int) -> bool:
        if metric > self.best:
            self.best = metric
            self.best_epoch = epoch
        return (epoch - self.best_epoch) > self.patience

################################################################################
#                          Model builder with real names                       #
################################################################################

def build_model(cfg: Dict[str, Any], in_dim: int, num_classes: int) -> nn.Module:
    name = cfg["name"].lower()
    p = cfg.get("params", {})
    if name == "gat":
        return GATNet(in_dim, p.get("hidden_dim", 256), num_classes,
                      num_layers=p.get("num_layers", 2), heads=p.get("heads", 4),
                      dropout=p.get("dropout", 0.4))
    elif name == "gatv2":
        return GATv2Net(in_dim, p.get("hidden_dim", 384), num_classes,
                        num_layers=p.get("num_layers", 4), heads=p.get("heads", 8),
                        dropout=p.get("dropout", 0.3))
    elif name == "graphtransformer":
        return GraphTransformerNet(in_dim, p.get("hidden_dim", 512), num_classes,
                                   num_layers=p.get("num_layers", 8),
                                   heads=p.get("heads", 4), dropout=p.get("dropout", 0.2))
    elif name == "geniepath":
        return GeniePathNet(in_dim, p.get("hidden_dim", 256), num_classes,
                            num_layers=p.get("num_layers", 3),
                            dropout=p.get("dropout", 0.3))
    else:
        raise ValueError(f"Unknown model name: {cfg['name']}")

################################################################################
#                                Train function                                #
################################################################################

def train(config: Dict[str, Any], run_name: str):
    seed_everything(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------- 1. DATA ------------------------------------
    data = load_dataset(config["dataset"], device)
    in_dim = data.x.shape[1]
    num_classes = int(data.y.max().item()) + 1

    # --------------------------- 2. MODEL -----------------------------------
    model = build_model(config["model"], in_dim, num_classes).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config["optim"]["lr"],
                             weight_decay=config["optim"].get("weight_decay", 0.0))
    loss_fn = nn.CrossEntropyLoss()

    # --------------------------- 3. SAMPLER ---------------------------------
    sampler = TADPSampler(
        data=data,
        reservoir_size=config["sampler"]["reservoir_size"],
        window=config["sampler"]["window"],
        device=device,
        lambdas=tuple(config["sampler"]["lambdas"]),
        mem_cap_gb=config["budget"].get("mem_cap_gb"),
        energy_cap_kwh=config["budget"].get("energy_cap_kwh"),
    )

    evaluator = Evaluator(data)
    energy_meter = EnergyMeter()
    stopper = EarlyStopper(patience=config["train"].get("patience", 50))

    max_epochs = config["train"]["epochs"]
    batch_size = config["train"].get("batch_size", 1024)

    history = {"train_loss": [], "val_metric": [], "epoch_energy_j": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        sampler.start_epoch()
        running_loss = 0.0
        batches = math.ceil(data.train_mask.sum().item() / batch_size)
        pbar = tqdm(range(batches), desc=f"[Epoch {epoch}]", leave=False)
        for _ in pbar:
            optim.zero_grad()
            idx = sampler.next_batch_indices(batch_size)
            out = model(data.x, data.edge_index)
            loss = loss_fn(out[idx], data.y[idx])
            loss.backward()
            optim.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        sampler.finish_epoch(energy_meter)
        history["train_loss"].append(running_loss / batches)
        history["epoch_energy_j"].append(energy_meter.epoch_energy_joules())

        model.eval()
        with torch.no_grad():
            val_metric = evaluator.eval(model, split="val", batch_size=4096)
        history["val_metric"].append(val_metric)
        print(f"Epoch {epoch:03d} | Loss {running_loss/batches:.4f} | Val {val_metric:.4f}")

        if stopper.step(val_metric, epoch):
            print("Early-stopping triggered.")
            break

    test_metric = evaluator.eval(model, split="test", batch_size=4096)
    print(f"TEST metric = {test_metric:.4f}")

    # ---------------------- save artefacts & figures -----------------------
    out_dir = Path(config["output"].get("dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history["train_loss"], "train_loss", out_dir)
    plot_training_curves(history["val_metric"], "val", out_dir, ylabel="Val-Metric")
    plot_metric_bars(history["epoch_energy_j"], "energy", out_dir, ylabel="Joules")

    res = {
        "run": run_name,
        "test_metric": test_metric,
        "best_val": max(history["val_metric"]),
        "epochs": len(history["val_metric"]),
        "energy_j": energy_meter.total_energy_joules(),
        "config": config,
    }
    with open(out_dir / f"{run_name}_results.json", "w") as fp:
        json.dump(res, fp, indent=2)

    return res
