"""
Training script for the bimodal Real-life Deception Detection model.

Uses 5-fold stratified cross-validation over the 121-sample dataset.

Usage (from project root):
    python -m deception_detection.real_life.train

    # Override defaults:
    python -m deception_detection.real_life.train \\
        --annotation_csv "Data/Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv" \\
        --transcript_dir "Data/Real-life_Deception_Detection_2016/Transcription" \\
        --checkpoint_dir checkpoints_real_life \\
        --folds 5 --epochs 200 --device cuda
"""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset

from ..data.sampler import make_weighted_sampler
from .collate import collate_fn
from .config import RealLifeConfig
from .dataset import RealLifeDataset
from .model import BimodalDeceptionModel


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])

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
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])

        total_loss += loss.item() * logits.size(0)
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    probs = [torch.sigmoid(torch.tensor(l)).item() for l in all_logits]
    binary_preds = [1 if p > 0.5 else 0 for p in probs]

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
        "logit_mean": float(np.mean(all_logits)),
        "logit_std": float(np.std(all_logits)),
    }


def run_cross_validation(config: RealLifeConfig):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[2]
    ckpt_dir = root / config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Load full dataset once (validates all transcript files at construction)
    full_dataset = RealLifeDataset(config, root)
    labels = full_dataset.get_labels()
    indices = list(range(len(full_dataset)))
    print(f"Dataset: {len(full_dataset)} samples  "
          f"(deceptive={labels.count(0)}, truthful={labels.count(1)})")

    if config.n_folds == 1:
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

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        train_labels = [labels[i] for i in train_idx]
        n_neg = train_labels.count(0)
        n_pos = train_labels.count(1)
        print(f"  Train: {len(train_labels)} samples | neg={n_neg} pos={n_pos}")

        train_sampler = make_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )

        model = BimodalDeceptionModel(config).to(device)
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
                model, train_loader, optimizer, scheduler, device, config.grad_clip,
            )
            val_metrics = evaluate(model, val_loader, device)

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

    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    for metric in ["accuracy", "f1", "auc_roc"]:
        values = [m.get(metric, 0) for m in fold_metrics]
        print(f"  {metric:12s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Train bimodal deception detector on Real-life dataset")
    parser.add_argument("--annotation_csv", default=None)
    parser.add_argument("--transcript_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()

    config = RealLifeConfig()
    if args.annotation_csv:
        config.annotation_csv = args.annotation_csv
    if args.transcript_dir:
        config.transcript_dir = args.transcript_dir
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

    print(f"Device:          {config.device}")
    print(f"Annotation CSV:  {config.annotation_csv}")
    print(f"Transcript dir:  {config.transcript_dir}")
    print(f"Checkpoint dir:  {config.checkpoint_dir}")
    print(f"Folds:           {config.n_folds}")
    print(f"Max epochs:      {config.max_epochs}")
    print(f"Batch size:      {config.batch_size}")
    print(f"Learning rate:   {config.learning_rate}")

    run_cross_validation(config)


if __name__ == "__main__":
    main()
