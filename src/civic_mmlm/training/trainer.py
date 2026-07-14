from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from civic_mmlm.data.types import modalities_from_batch
from civic_mmlm.models.constraints import DualVariables
from civic_mmlm.utils.io import save_checkpoint, save_json

from .losses import UnifiedObjective


def move_batch_to_device(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return value
    return value


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: dict,
        device: torch.device,
        output_dir: str | Path,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_cfg = config["training"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 2e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.objective = UnifiedObjective(config)
        self.duals = DualVariables().to(device)
        self.dual_lr = float(train_cfg.get("dual_learning_rate", 0.03))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    def _forward(self, batch: dict, modalities_key: str, certificate: bool) -> dict:
        return self.model(
            modalities_from_batch(batch, modalities_key),
            legality=batch["legality"],
            budget=batch["budget"],
            action_costs=batch["action_costs"],
            certificate_action=batch["label"] if certificate else None,
            compute_certificate=certificate,
        )

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals: dict[str, float] = defaultdict(float)
        count = 0
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output = self._forward(batch, "modalities", certificate=True)
            valid_output = self._forward(batch, "valid_modalities", certificate=False)
            material_output = self._forward(batch, "material_modalities", certificate=False)
            violations = {
                "constraint": output["decision"].constraint_slack.mean(),
                "fairness": output["decision"].probabilities.new_zeros(()),
            }
            dual_penalty = self.duals.penalty(violations)
            loss_output = self.objective(
                output, valid_output, material_output, batch, dual_penalty
            )
            loss_output.total.backward()
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.duals.projected_ascent(loss_output.violations, self.dual_lr)

            batch_size = batch["label"].shape[0]
            count += batch_size
            totals["total"] += float(loss_output.total.detach().item()) * batch_size
            for name, value in loss_output.components.items():
                totals[name] += float(value.detach().item()) * batch_size
        return {name: value / max(count, 1) for name, value in totals.items()}

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        losses = 0.0
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, self.device)
            output = self._forward(batch, "modalities", certificate=True)
            valid_output = self._forward(batch, "valid_modalities", certificate=False)
            material_output = self._forward(batch, "material_modalities", certificate=False)
            loss_output = self.objective(
                output, valid_output, material_output, batch, dual_penalty=None
            )
            predictions = output["decision"].probabilities.argmax(-1)
            correct += int((predictions == batch["label"]).sum().item())
            total += int(batch["label"].numel())
            losses += float(loss_output.total.item()) * batch["label"].shape[0]
        return {"loss": losses / max(total, 1), "accuracy": correct / max(total, 1)}

    def fit(self, train_loader: DataLoader, dev_loader: DataLoader) -> list[dict]:
        epochs = int(self.config["training"].get("epochs", 5))
        history = []
        best_accuracy = -1.0
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            dev_metrics = self.validate(dev_loader)
            record = {"epoch": epoch, "train": train_metrics, "dev": dev_metrics}
            history.append(record)
            print(
                f"epoch={epoch:02d} train_loss={train_metrics['total']:.4f} "
                f"dev_loss={dev_metrics['loss']:.4f} dev_acc={dev_metrics['accuracy']:.4f}"
            )
            if dev_metrics["accuracy"] > best_accuracy:
                best_accuracy = dev_metrics["accuracy"]
                save_checkpoint(
                    self.model,
                    self.config,
                    self.output_dir / "best_model.pt",
                    epoch=epoch,
                    dev_metrics=dev_metrics,
                )
        save_json(history, self.output_dir / "training_history.json")
        return history
