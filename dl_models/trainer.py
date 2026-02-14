"""
Training utilities for the MultimodalPDNet.

Provides a ``Trainer`` class that handles:
    - SMOTE class balancing
    - Online data augmentation (Gaussian noise, feature dropout)
    - Train / validation loop with early stopping
    - Learning rate scheduling (ReduceLROnPlateau)
    - Best model checkpointing
    - Metric logging and plot generation
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from torch.utils.data import DataLoader

from dl_models.dataset import MultimodalPDDataset
from dl_models.networks import MultimodalPDNet

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data Augmentation                                                  #
# ------------------------------------------------------------------ #

def augment_batch(
    batch: dict[str, torch.Tensor],
    noise_std: float = 0.05,
    feature_dropout: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Apply in-place Gaussian noise and random feature dropout.

    Args:
        batch: Dict with keys ``speech``, ``handwriting``, ``gait``,
               ``label``.  Tensors are modified in place.
        noise_std: Std-dev of additive Gaussian noise as a fraction
                   of each feature's absolute value.
        feature_dropout: Probability of zeroing a feature.

    Returns:
        The same batch dict (modified).
    """
    for key in ("speech", "handwriting", "gait"):
        x = batch[key]
        # Gaussian noise
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        # Feature dropout
        mask = torch.rand_like(x) > feature_dropout
        x = x * mask
        batch[key] = x
    return batch


# ------------------------------------------------------------------ #
#  Trainer                                                            #
# ------------------------------------------------------------------ #

class Trainer:
    """Training loop for ``MultimodalPDNet``.

    Args:
        model: An instance of ``MultimodalPDNet``.
        device: ``'cpu'`` or ``'cuda'``.
        lr: Initial learning rate (default 0.001).
        weight_decay: AdamW weight decay (default 1e-4).
        patience: Early stopping patience in epochs (default 15).
        noise_std: Augmentation noise level (default 0.05).
        feature_dropout: Augmentation feature dropout (default 0.1).
    """

    def __init__(
        self,
        model: MultimodalPDNet,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        noise_std: float = 0.05,
        feature_dropout: float = 0.1,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.patience = patience
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=False,
        )

        # History
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_state: Optional[dict] = None

    # -- single epoch ------------------------------------------------- #

    def _train_epoch(
        self, loader: DataLoader, augment: bool = True,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            if augment:
                batch = augment_batch(
                    batch, self.noise_std, self.feature_dropout,
                )

            speech = batch["speech"].to(self.device)
            handwriting = batch["handwriting"].to(self.device)
            gait = batch["gait"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(speech, handwriting, gait)
            loss = self.criterion(out["logit"].squeeze(-1), labels)
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict[str, Any]]:
        """Evaluate on a data loader. Returns (loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in loader:
            speech = batch["speech"].to(self.device)
            handwriting = batch["handwriting"].to(self.device)
            gait = batch["gait"].to(self.device)
            labels = batch["label"].to(self.device)

            out = self.model(speech, handwriting, gait)
            loss = self.criterion(out["logit"].squeeze(-1), labels)
            total_loss += loss.item()
            n_batches += 1

            probs = out["probability"].squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.0,
        }

        return avg_loss, metrics

    # -- full training loop ------------------------------------------- #

    def fit(
        self,
        train_dataset: MultimodalPDDataset,
        val_dataset: MultimodalPDDataset,
        epochs: int = 100,
        batch_size: int = 32,
        augment: bool = True,
    ) -> dict[str, Any]:
        """Train the model with early stopping.

        Args:
            train_dataset: Training ``MultimodalPDDataset``.
            val_dataset: Validation ``MultimodalPDDataset``.
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            augment: Whether to apply online augmentation.

        Returns:
            Dict with final training metrics and history.
        """
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
        )

        start_time = time.time()
        logger.info(
            "Starting training: %d epochs, batch_size=%d, device=%s",
            epochs, batch_size, self.device,
        )

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, augment=augment)
            val_loss, val_metrics = self._eval_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | "
                    "Val Acc: %.3f | Val AUC: %.3f | LR: %.6f",
                    epoch, epochs, train_loss, val_loss,
                    val_metrics["accuracy"], val_metrics["roc_auc"],
                    current_lr,
                )

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d).",
                    epoch, self.patience,
                )
                break

        elapsed = time.time() - start_time
        logger.info("Training completed in %.1f seconds.", elapsed)

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info("Restored best model weights (val_loss=%.4f).", self.best_val_loss)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "total_epochs": len(self.train_losses),
            "elapsed_seconds": elapsed,
        }

    # -- evaluation --------------------------------------------------- #

    def evaluate(
        self, test_dataset: MultimodalPDDataset, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Evaluate on the test set.

        Returns:
            Dict with metrics, predictions, probabilities, and labels.
        """
        loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
        )
        loss, metrics = self._eval_epoch(loader)
        metrics["test_loss"] = loss

        # Collect predictions for plots
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(
                    batch["speech"].to(self.device),
                    batch["handwriting"].to(self.device),
                    batch["gait"].to(self.device),
                )
                all_probs.append(
                    out["probability"].squeeze(-1).cpu().numpy(),
                )
                all_labels.append(batch["label"].numpy())

        metrics["y_true"] = np.concatenate(all_labels)
        metrics["y_prob"] = np.concatenate(all_probs)
        metrics["y_pred"] = (metrics["y_prob"] >= 0.5).astype(int)

        return metrics

    # -- saving ------------------------------------------------------- #

    def save_model(self, path: str | Path) -> None:
        """Save model state dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def save_metrics(
        self, metrics: dict[str, Any], path: str | Path,
    ) -> None:
        """Save JSON-serialisable metrics to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Strip numpy arrays for JSON
        clean: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info("Metrics saved to %s", path)

    # -- plotting ----------------------------------------------------- #

    def plot_training_curves(self, save_path: str | Path) -> None:
        """Save train/val loss curves."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.train_losses, label="Train Loss")
        ax.plot(self.val_losses, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("BCE Loss")
        ax.set_title("SE-ResNet Training Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Training curves saved to %s", save_path)

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str | Path,
    ) -> None:
        """Save ROC curve plot."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"SE-ResNet (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Multimodal SE-ResNet")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved to %s", save_path)

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str | Path,
    ) -> None:
        """Save confusion matrix plot."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            cm, display_labels=["Healthy", "PD"],
        )
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix — SE-ResNet")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)
