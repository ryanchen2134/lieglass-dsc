"""
Training script v2 — PECL-Adapted 3-Tower architecture.

Key differences from v1:
  - GroupKFold on speaker ID (sample_id.split('_')[0]) prevents identity leakage.
  - CosineAnnealingLR replaces OneCycleLR (consistent LR throughout training).
  - CrossEntropyLoss (2-class) replaces BCEWithLogitsLoss.
  - Model returns (B, 2) logits; predictions use argmax / softmax[:, 1].

Usage (from project root):
    python -m deception_detection.train \
        --manifest Data/manifest_dolos.csv \
        --feature_dir features \
        --checkpoint_dir checkpoints
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pathlib import Path

from .config import ModelConfig
from .data.dataset import DeceptionDataset
from .data.collate import collate_fn
from .data.sampler import make_weighted_sampler
from .models.full_model import MultimodalDeceptionModel


def train_one_epoch(model, loader, optimizer, loss_fn, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        logits = model(batch)                              # (B, 2)
        loss = loss_fn(logits, batch["label"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * logits.size(0)
        probs = F.softmax(logits.detach(), dim=-1)[:, 1]  # P(deceptive)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    binary_preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, binary_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        logits = model(batch)                              # (B, 2)
        loss = loss_fn(logits, batch["label"])

        total_loss += loss.item() * logits.size(0)
        probs = F.softmax(logits, dim=-1)[:, 1]            # P(deceptive)
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    binary_preds = [1 if p > 0.5 else 0 for p in all_probs]

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    try:
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
    except Exception:
        f1 = 0.0

    return {
        "loss":     avg_loss,
        "accuracy": accuracy_score(all_labels, binary_preds),
        "f1":       f1,
        "auc_roc":  auc,
        "prob_mean": float(np.mean(all_probs)),
        "prob_std":  float(np.std(all_probs)),
    }


def run_cross_validation(config: ModelConfig):
    device = torch.device(config.device)
    root = Path(__file__).resolve().parents[1]
    manifest   = root / config.manifest_csv
    feature_dir = root / config.feature_dir
    ckpt_dir   = root / config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset to get labels and speaker group IDs
    full_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=False)
    labels  = full_dataset.get_labels()
    indices = list(range(len(full_dataset)))

    # --------------------------------------------------------------------- #
    # Identity-aware splits: group by speaker (first token of sample_id).   #
    # E.g. "AN_WILTY_EP15_lie10" → group "AN".                              #
    # GroupKFold ensures no speaker appears in both train and val.           #
    # --------------------------------------------------------------------- #
    if config.n_folds == 1:
        # Smoke-test mode: simple 80/20 stratified split (no group guarantee)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                     random_state=config.seed)
        splits = list(sss.split(indices, labels))
    else:
        groups = [s.split("_")[0] for s, _ in full_dataset.samples]
        gkf = GroupKFold(n_splits=config.n_folds)
        splits = list(gkf.split(indices, labels, groups=groups))

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{len(splits)}")
        print(f"{'='*60}")

        # Fold-specific datasets
        train_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=True)
        train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
        val_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=False)
        val_dataset.samples = [val_dataset.samples[i] for i in val_idx]

        train_labels = [labels[i] for i in train_idx]
        n_neg = train_labels.count(0)
        n_pos = train_labels.count(1)
        print(f"  Train: {len(train_labels)} samples | neg={n_neg} pos={n_pos} "
              f"(balanced via sampler)")

        train_sampler = make_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # ----------------------------------------------------------------- #
        # Model, optimizer, scheduler, loss                                  #
        # ----------------------------------------------------------------- #
        model = MultimodalDeceptionModel(config).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"  Trainable parameters: {n_trainable:,}")

        optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        # CosineAnnealingLR: smooth decay from lr → eta_min over max_epochs.
        # step() called once per epoch (not per batch).
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.max_epochs,
            eta_min=1e-5,
        )
        loss_fn = nn.CrossEntropyLoss()

        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(config.max_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, loss_fn,
                device, config.grad_clip,
            )
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            scheduler.step()   # cosine decay — one step per epoch

            print(
                f"  Epoch {epoch+1:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1={val_metrics['f1']:.3f} "
                f"val_auc={val_metrics['auc_roc']:.3f} "
                f"p_μ={val_metrics['prob_mean']:.3f} "
                f"p_σ={val_metrics['prob_std']:.3f}"
            )

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_dir / f"fold_{fold}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        print(f"\n  Fold {fold+1} best: "
              f"auc={best_metrics.get('auc_roc', 0):.3f} "
              f"f1={best_metrics.get('f1', 0):.3f} "
              f"acc={best_metrics.get('accuracy', 0):.3f}")
        fold_metrics.append(best_metrics)

    # Aggregate results
    print("\n" + "="*60)
    print("Cross-Validation Results (GroupKFold, identity-aware)")
    print("="*60)
    for metric in ["accuracy", "f1", "auc_roc"]:
        values = [m.get(metric, 0) for m in fold_metrics]
        print(f"  {metric:12s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train LieGlass-DSC v2 (PECL-adapted 3-tower)"
    )
    parser.add_argument("--manifest",       default=None)
    parser.add_argument("--feature_dir",    default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device",         default=None)
    parser.add_argument("--batch_size",     type=int, default=None)
    parser.add_argument("--epochs",         type=int, default=None)
    parser.add_argument("--folds",          type=int, default=None)
    parser.add_argument("--lr",             type=float, default=None)
    args = parser.parse_args()

    config = ModelConfig()
    if args.manifest:       config.manifest_csv    = args.manifest
    if args.feature_dir:    config.feature_dir     = args.feature_dir
    if args.checkpoint_dir: config.checkpoint_dir  = args.checkpoint_dir
    if args.device:         config.device          = args.device
    if args.batch_size:     config.batch_size      = args.batch_size
    if args.epochs:         config.max_epochs      = args.epochs
    if args.folds:          config.n_folds         = args.folds
    if args.lr:             config.learning_rate   = args.lr

    print(f"Device:    {config.device}")
    print(f"Manifest:  {config.manifest_csv}")
    print(f"Features:  {config.feature_dir}")
    print(f"LR:        {config.learning_rate}")
    print(f"Folds:     {config.n_folds}")

    run_cross_validation(config)


if __name__ == "__main__":
    main()
