from typing import Dict, List

import numpy as np
import torch
import networkx as nx
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import add_self_loops, coalesce
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx

# -----------------------------------------------------------------------------
# Fallback synthetic graph for smoke-tests (unchanged)
# -----------------------------------------------------------------------------

def _create_synthetic(num_nodes: int, num_edges: int, in_dim: int, num_classes: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    g = nx.gnm_random_graph(num_nodes, num_edges, seed=seed, directed=False)
    data = from_networkx(g)
    data.x = torch.tensor(rng.standard_normal(size=(num_nodes, in_dim)), dtype=torch.float)
    data.y = torch.tensor(rng.integers(0, num_classes, size=num_nodes), dtype=torch.long)
    data.edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = coalesce(data.edge_index)

    idx = rng.permutation(num_nodes)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.2 * num_nodes)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[torch.tensor(train_idx)] = True
    data.val_mask[torch.tensor(val_idx)] = True
    data.test_mask[torch.tensor(test_idx)] = True
    return data

# -----------------------------------------------------------------------------
# Generic helper for Hugging Face graph datasets
# -----------------------------------------------------------------------------

def _tensor_from_first_available(d: dict, keys: List[str], dtype):
    for k in keys:
        if k in d:
            return torch.tensor(d[k], dtype=dtype)
    raise KeyError(f"None of the keys {keys} found in dict")


def _build_masks(num_nodes: int, graph_dict: dict):
    train_mask = _tensor_from_first_available(graph_dict, ["train_mask", "train_mask_bool"], torch.bool) if any(k in graph_dict for k in ["train_mask", "train_mask_bool"]) else None
    val_mask = _tensor_from_first_available(graph_dict, ["val_mask", "valid_mask"], torch.bool) if any(k in graph_dict for k in ["val_mask", "valid_mask"]) else None
    test_mask = _tensor_from_first_available(graph_dict, ["test_mask"], torch.bool) if "test_mask" in graph_dict else None

    if train_mask is None or val_mask is None or test_mask is None:
        # default 60/20/20 split if masks are missing
        rng = np.random.default_rng(0)
        idx = rng.permutation(num_nodes)
        n_train = int(0.6 * num_nodes)
        n_val = int(0.2 * num_nodes)
        train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[torch.tensor(train_idx)] = True
        val_mask[torch.tensor(val_idx)] = True
        test_mask[torch.tensor(test_idx)] = True
    return train_mask, val_mask, test_mask


def _load_hf_graph_dataset(hf_id: str):
    ds = load_dataset(hf_id)
    if "train" in ds:
        graph_dict = ds["train"][0]  # single graph per dataset
    else:
        # If dataset not split, grab the first split arbitrarily
        first_split = list(ds.keys())[0]
        graph_dict = ds[first_split][0]

    x = _tensor_from_first_available(graph_dict, ["x", "node_features", "features", "feat"], torch.float)
    y = _tensor_from_first_available(graph_dict, ["y", "labels", "label"], torch.long)

    if "edge_index" in graph_dict:
        edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    elif "edge_src" in graph_dict and "edge_dst" in graph_dict:
        edge_index = torch.vstack([
            torch.tensor(graph_dict["edge_src"], dtype=torch.long),
            torch.tensor(graph_dict["edge_dst"], dtype=torch.long)
        ])
    else:
        raise KeyError("Edge information not found in dataset")

    data = Data(x=x, y=y, edge_index=edge_index)
    train_mask, val_mask, test_mask = _build_masks(data.num_nodes, graph_dict)
    data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask

    data.edge_index, _ = add_self_loops(data.edge_index)
    data.edge_index = coalesce(data.edge_index)
    return data

# -----------------------------------------------------------------------------
# MAIN PRE-PROCESS DISPATCHER – now contains real dataset loading logic
# -----------------------------------------------------------------------------

def preprocess_data(cfg: Dict):
    d_cfg = cfg["dataset"]
    name = d_cfg["name"].upper()

    if name == "SYNTHETIC_SMOKE":
        data = _create_synthetic(num_nodes=d_cfg["num_nodes"],
                                 num_edges=d_cfg["num_edges"],
                                 in_dim=cfg["model"]["in_channels"],
                                 num_classes=cfg["model"]["num_classes"],
                                 seed=cfg["evaluation"].get("seed", 1))

    elif name in {"REDDIT_COMMENTS", "REDDIT"}:
        data = _load_hf_graph_dataset("HuggingFaceGECLM/REDDIT_comments")

    elif name == "TEXAS":
        data = _load_hf_graph_dataset("Allen-UQ/texas_all_nodes")

    elif name == "WISCONSIN":
        data = _load_hf_graph_dataset("Allen-UQ/wisconsin_all_nodes")

    else:
        raise NotImplementedError(f"Dataset {name} not integrated.  NO-FALLBACK policy active.")

    # --------  Feature normalisation  ---------
    scaler = StandardScaler()
    data.x = torch.tensor(scaler.fit_transform(data.x), dtype=torch.float)

    # Update config with actual dimensions (helps when placeholder values were used)
    cfg["model"]["in_channels"] = data.num_node_features
    cfg["model"]["num_classes"] = int(data.y.max().item() + 1)
    return data