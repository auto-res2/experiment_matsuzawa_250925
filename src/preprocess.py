"""src/preprocess.py
Concrete dataset loaders for OGB-Products / Papers100M and Friendster.
"""
from __future__ import annotations
from typing import Dict, Any

import torch, numpy as np
from torch_geometric.data import Data
from torch_geometric.datasets import SNAPDataset
from ogb.nodeproppred import PygNodePropPredDataset


def _ogb_loader(name: str, root: str) -> Data:
    dset = PygNodePropPredDataset(name=name, root=root)
    split = dset.get_idx_split()
    data = dset[0]
    data.y = data.y.squeeze().to(torch.long)
    # build boolean masks expected by the rest of the code
    num_nodes = data.y.size(0)
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[torch.from_numpy(split['train'])] = True
    data.val_mask  [torch.from_numpy(split['valid'])] = True
    data.test_mask [torch.from_numpy(split['test'])]  = True
    return data


def _friendster_loader(root: str) -> Data:
    ds = SNAPDataset(root=root, name="soc-friendster")
    data = ds[0]
    # No features – use degree + 1 as scalar feature
    deg = torch.bincount(data.edge_index[0])
    if deg.size(0) < data.num_nodes:
        deg = torch.cat([deg, deg.new_zeros(data.num_nodes - deg.size(0))])
    data.x = torch.log1p(deg).unsqueeze(1).to(torch.float32)
    # Simple chronological (by node id) 70/10/20 split
    num_nodes = data.num_nodes
    idx = torch.arange(num_nodes)
    data.train_mask = idx < int(0.7 * num_nodes)
    data.val_mask   = (idx >= int(0.7 * num_nodes)) & (idx < int(0.8 * num_nodes))
    data.test_mask  = idx >= int(0.8 * num_nodes)
    data.y = torch.randint(0, 2, (num_nodes,), dtype=torch.long)  # dummy labels
    return data


def load_dataset(cfg: Dict[str, Any], device) -> Data:
    name = cfg["name"].lower()
    root = cfg.get("root", "data")
    if name in {"ogbn-products", "ogbn_products"}:
        data = _ogb_loader("ogbn-products", root)
    elif name in {"ogbn-papers100m", "ogbn_papers100m"}:
        data = _ogb_loader("ogbn-papers100M", root)
    elif name in {"friendster", "soc-friendster"}:
        data = _friendster_loader(root)
    else:
        raise ValueError(f"Unknown dataset {cfg['name']}")
    return data.to(device)
