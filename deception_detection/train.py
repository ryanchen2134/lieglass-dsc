"""
Training script with 5-fold stratified cross-validation.

Usage (from project root):
    python -m deception_detection.train \
        --manifest Data/manifest_dolos.csv \
        --feature_dir features \
        --checkpoint_dir checkpoints

Or use default paths from config.py.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from pathlib import Path

from .config import ModelConfig
from .data.dataset import DeceptionDataset
from .data.collate import collate_fn
from .data.sampler import make_weighted_sampler
from .models.fusion_model import FusionModel


def train_one_epoch(model, loader, optimizer, scheduler, pos_weight, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = model(batch)  # (B,)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            batch["label"],
            pos_weight=pos_weight.to(device),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * logits.size(0)
        preds = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, pos_weight, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            batch["label"],
            pos_weight=pos_weight.to(device),
        )

        total_loss += loss.item() * logits.size(0)
        all_preds.extend(logits.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    # Convert raw logits → probabilities for metrics
    probs = [torch.sigmoid(torch.tensor(l)).item() for l in all_preds]
    binary_preds = [1 if p > 0.5 else 0 for p in probs]

    # Guard against single-class batches in early training
    try:
        auc = roc_auc_score(all_labels, probs)
    except ValueError:
        auc = 0.5
    try:
        f1 = f1_score(all_labels, binary_preds, zero_division=0)
    except Exception:
        f1 = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, binary_preds),
        "f1": f1,
        "auc_roc": auc,
        "logit_mean": float(np.mean(all_preds)),
        "logit_std": float(np.std(all_preds)),
    }


def run_cross_validation(config: ModelConfig):
    device = torch.device(config.device)
    root = Path(__file__).resolve().parents[1]
    manifest = root / config.manifest_csv
    feature_dir = root / config.feature_dir
    ckpt_dir = root / config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset for CV splitting
    full_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=False)
    labels = full_dataset.get_labels()
    indices = list(range(len(full_dataset)))

    if config.n_folds == 1:
        # Smoke-test mode: single 80/20 stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        splits = list(sss.split(indices, labels))
    else:
        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
        splits = list(skf.split(indices, labels))
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

        # Class balance info — rebalancing handled entirely by the weighted sampler
        train_labels = [labels[i] for i in train_idx]
        n_neg = train_labels.count(0)
        n_pos = train_labels.count(1)
        pos_weight = torch.tensor([1.0], dtype=torch.float32)  # sampler already balances classes
        print(f"  Train: {len(train_labels)} samples | neg={n_neg} pos={n_pos} (balanced via sampler)")

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

        # Model, optimizer, scheduler
        model = FusionModel(config).to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = config.max_epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(config.max_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                pos_weight, device, config.grad_clip,
            )
            val_metrics = evaluate(model, val_loader, pos_weight, device)

            print(
                f"  Epoch {epoch+1:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1={val_metrics['f1']:.3f} val_auc={val_metrics['auc_roc']:.3f} "
                f"logit_μ={val_metrics['logit_mean']:+.2f} logit_σ={val_metrics['logit_std']:.2f}"
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

        print(f"\n  Fold {fold+1} best: auc={best_metrics.get('auc_roc', 0):.3f} "
              f"f1={best_metrics.get('f1', 0):.3f} acc={best_metrics.get('accuracy', 0):.3f}")
        fold_metrics.append(best_metrics)

    # Aggregate
    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    for metric in ["accuracy", "f1", "auc_roc"]:
        values = [m.get(metric, 0) for m in fold_metrics]
        print(f"  {metric:12s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()

    config = ModelConfig()
    if args.manifest:
        config.manifest_csv = args.manifest
    if args.feature_dir:
        config.feature_dir = args.feature_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.max_epochs = args.epochs
    if args.folds:
        config.n_folds = args.folds

    print(f"Device: {config.device}")
    print(f"Manifest: {config.manifest_csv}")
    print(f"Features: {config.feature_dir}")

    run_cross_validation(config)


if __name__ == "__main__":
    main()
