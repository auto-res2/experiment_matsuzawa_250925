import os
import json
import time
import yaml
from datetime import datetime
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from .preprocess import create_datasets, build_transforms
from .evaluate import Evaluator
from .flash_tta import FlashTTA

__all__ = ["Trainer"]

def _instantiate_backbone(model_cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate a timm backbone from the config.
    Supports any model available in timm (including HF-mirrored weights, e.g.  ``resnet50.a1_in1k``).
    The function automatically replaces the classifier head so that the backbone
    always returns logits of ``num_classes``.
    """
    import timm  # local import keeps CLI start-up fast when timm isn’t needed

    model_name: str = model_cfg.get("name", "resnet18")
    pretrained: bool = bool(model_cfg.get("pretrained", False))
    num_classes: int = int(model_cfg.get("num_classes", 1000))

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # Some timm models expose classifier under different names (fc / head / classifier)
    # but we already asked timm to build the proper final Linear layer with num_classes.
    return model


def _wrap_with_method(model: nn.Module, method_cfg: Dict[str, Any]) -> nn.Module:
    """Wrap ``model`` with the adaptation method selected in the YAML config."""
    method_name = method_cfg.get("method", "source").lower()

    if method_name == "source":
        return model  # no adaptation at all

    if method_name == "bn_adapt":
        # Simple BN statistics recomputation at test-time
        model.train()  # keep BN layers in train mode so that running stats update
        for p in model.parameters():
            p.requires_grad = False
        return model

    if method_name == "flash_tta":
        proj_dim = int(method_cfg.get("proj_dim", 16))
        gru_hidden = int(method_cfg.get("gru_hidden", 256))
        return FlashTTA(model, proj_dim=proj_dim, gru_hidden=gru_hidden)

    # If you reach this line the method is unknown
    raise NotImplementedError(f"Test-time adaptation method '{method_name}' is not implemented")


class Trainer:
    """End-to-end training / evaluation driver.

    • Creates datasets & loaders
    • Builds backbone + adaptation wrapper
    • (Optionally) runs supervised training
    • Evaluates on the requested test split and dumps results/figures
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----------------------------- DATASETS ------------------------------ #
        tf_train, tf_val = build_transforms(cfg)
        (self.train_set, self.val_set, self.test_set) = create_datasets(cfg, tf_train, tf_val)

        train_bs = int(cfg["training"].get("batch_size", 128))
        eval_bs = int(cfg["evaluation"].get("batch_size", 256))

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=train_bs,
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 8)),
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 8)),
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=int(cfg.get("num_workers", 8)),
            pin_memory=True,
        )

        # ---------------------------- MODEL ---------------------------------- #
        backbone = _instantiate_backbone(cfg["model"])
        self.model = _wrap_with_method(backbone, cfg["model"]).to(self.device)

        # -------------------------- OPTIMISER -------------------------------- #
        optim_name = cfg["training"].get("optimizer", "adam").lower()
        lr = float(cfg["training"].get("lr", 1e-3))
        wd = float(cfg["training"].get("weight_decay", 1e-4))

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if optim_name == "adam":
            self.optimizer = Adam(trainable_params, lr=lr, weight_decay=wd)
        elif optim_name == "sgd":
            self.optimizer = SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimiser {optim_name}")

        total_epochs = int(cfg["training"].get("epochs", 0))
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_epochs, 1))

        self.criterion = nn.CrossEntropyLoss()

        # ----------------------------- I/O ----------------------------------- #
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(cfg.get("output_dir", "runs"), cfg.get("experiment_id", timestamp))
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_acc = 0.0
        self.results: Dict[str, Any] = {
            "train_loss_per_epoch": [],
            "val_acc_per_epoch": [],
            "timestamp": timestamp,
            "config": cfg,
        }

    # --------------------------- TRAINING UTILS ----------------------------- #
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss, total, correct = 0.0, 0, 0
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
        avg_loss = running_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)
        print(f"[Epoch {epoch}] Train loss {avg_loss:.4f} | Acc {train_acc:.2f}%")
        return avg_loss

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        correct, total = 0, 0
        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            _, preds = logits.max(1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
        acc = 100.0 * correct / max(total, 1)
        print(f"[Epoch {epoch}] Val Acc {acc:.2f}%")
        return acc

    # ------------------------------ DRIVER ---------------------------------- #
    def run(self) -> Dict[str, Any]:
        epochs = int(self.cfg["training"].get("epochs", 0))
        if epochs > 0:
            for epoch in range(1, epochs + 1):
                loss = self._train_one_epoch(epoch)
                acc = self._validate(epoch)
                self.scheduler.step()
                self.results["train_loss_per_epoch"].append(loss)
                self.results["val_acc_per_epoch"].append(acc)

                if acc > self.best_acc:
                    self.best_acc = acc
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, "best_model.pt"))
        else:
            print("Epochs == 0 → skipping supervised training (using pre-trained weights)")

        # -------------------------- FINAL TEST ----------------------------- #
        evaluator = Evaluator(self.cfg, self.device)
        test_metrics = evaluator.evaluate(self.model, self.test_loader)
        self.results["test_metrics"] = test_metrics

        json.dump(self.results, open(os.path.join(self.output_dir, "results.json"), "w"), indent=4)

        # Pretty print for terminal
        print("\n===== EXPERIMENT SUMMARY =====")
        print(json.dumps(self.results, indent=4))
        print("==============================\n")

        return self.results
