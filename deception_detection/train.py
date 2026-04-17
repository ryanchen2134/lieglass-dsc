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
import threading
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


class _GPUMonitor:
    """
    Background thread that samples GPU utilisation and VRAM every `interval`
    seconds via pynvml (falls back to torch VRAM-only if pynvml is absent).
    Call reset_epoch() before each epoch and epoch_summary() after it.
    """

    def __init__(self, interval: int = 10):
        self.interval  = interval
        self._n_gpus   = torch.cuda.device_count()
        self._samples: list[dict] = []
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self._n_gpus > 0:
            self._thread.start()

    def stop(self):
        self._stop.set()

    def reset_epoch(self):
        with self._lock:
            self._samples.clear()

    def epoch_summary(self) -> str:
        with self._lock:
            samples = list(self._samples)
        if not samples:
            return ""
        avg_util = [sum(s["util"][i] for s in samples) / len(samples) for i in range(self._n_gpus)]
        avg_mem  = [sum(s["mem"][i]  for s in samples) / len(samples) for i in range(self._n_gpus)]
        import math
        util_str = "/".join("--" if math.isnan(u) else f"{u:.0f}%" for u in avg_util)
        mem_str  = "/".join(f"{m:.1f}G" for m in avg_mem)
        return f"gpu={util_str} vram={mem_str}"

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self._n_gpus)]
            while not self._stop.wait(self.interval):
                util = [pynvml.nvmlDeviceGetUtilizationRates(h).gpu for h in handles]
                mem  = [pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 ** 3 for h in handles]
                with self._lock:
                    self._samples.append({"util": util, "mem": mem})
        except ImportError:
            # pynvml unavailable — report VRAM from torch (util shown as --)
            while not self._stop.wait(self.interval):
                mem = [torch.cuda.memory_reserved(i) / 1024 ** 3 for i in range(self._n_gpus)]
                with self._lock:
                    self._samples.append({"util": [float("nan")] * self._n_gpus, "mem": mem})
        except Exception:
            pass


def train_one_epoch(model, loader, optimizer, scheduler, pos_weight, device,
                    grad_clip, grad_accum_steps=1, scaler=None, label_smoothing=0.0):
    """
    Train for one epoch with optional AMP (scaler) and gradient accumulation.

    grad_accum_steps > 1: gradients are accumulated across that many mini-batches
    before a parameter update, giving an effective batch size of
    batch_size × grad_accum_steps with lower peak memory per step.
    """
    model.train()
    use_amp    = scaler is not None
    device_str = device.type   # "cuda" or "cpu"
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.autocast(device_type=device_str, enabled=use_amp):
            logits = model(batch)   # (B,)
            targets = batch["label"]
            if label_smoothing > 0.0:
                targets = targets * (1 - label_smoothing) + 0.5 * label_smoothing
            loss   = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight.to(device),
            )
            # Scale loss for gradient accumulation so effective LR is unchanged
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Unscale + update every grad_accum_steps batches (or on last batch)
        is_update_step = ((step + 1) % grad_accum_steps == 0
                          or (step + 1) == len(loader))
        if is_update_step:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Track unscaled loss (multiply back)
        total_loss += loss.item() * grad_accum_steps * logits.size(0)
        preds = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
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
    full_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=False, n_frames=config.n_frames)
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
    monitor = _GPUMonitor(interval=10)
    monitor.start()

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*60}", flush=True)
        print(f"Fold {fold + 1}/{len(splits)}", flush=True)
        print(f"{'='*60}", flush=True)

        # Fold-specific datasets
        train_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=True, n_frames=config.n_frames)
        train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
        val_dataset = DeceptionDataset(str(manifest), str(feature_dir), augment=False, n_frames=config.n_frames)
        val_dataset.samples = [val_dataset.samples[i] for i in val_idx]

        # Class balance info — rebalancing handled entirely by the weighted sampler
        train_labels = [labels[i] for i in train_idx]
        n_neg = train_labels.count(0)
        n_pos = train_labels.count(1)
        pos_weight = torch.tensor([1.0], dtype=torch.float32)  # sampler already balances classes
        print(f"  Train: {len(train_labels)} samples | neg={n_neg} pos={n_pos} (balanced via sampler)", flush=True)

        train_sampler = make_weighted_sampler(train_labels)

        pin = device.type == "cuda"
        # prefetch_factor is only valid when num_workers > 0
        extra = {"prefetch_factor": config.prefetch_factor} if config.num_workers > 0 else {}
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=config.num_workers > 0,
            drop_last=True,
            **extra,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.num_workers,
            pin_memory=pin,
            persistent_workers=config.num_workers > 0,
            **extra,
        )

        # Model, optimizer, scheduler, AMP scaler
        model = FusionModel(config).to(device)
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)", flush=True)
            model = torch.nn.DataParallel(model)
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)", flush=True)
        optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)

        # OneCycleLR counts parameter updates, not raw batches.
        # With gradient accumulation, one update happens every grad_accum_steps batches.
        updates_per_epoch = max(1, len(train_loader) // config.grad_accum_steps)
        total_steps = config.max_epochs * updates_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
        )

        # Automatic Mixed Precision — enabled only on CUDA; no-ops on CPU
        scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

        best_val_auc = 0.0
        best_metrics = {}
        patience_counter = 0

        for epoch in range(config.max_epochs):
            monitor.reset_epoch()
            print(f"  Epoch {epoch+1}/{config.max_epochs}", flush=True)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                pos_weight, device, config.grad_clip,
                grad_accum_steps=config.grad_accum_steps,
                scaler=scaler,
                label_smoothing=config.label_smoothing,
            )
            val_metrics = evaluate(model, val_loader, pos_weight, device)
            gpu_info = monitor.epoch_summary()

            print(
                f"  Epoch {epoch+1:3d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1={val_metrics['f1']:.3f} val_auc={val_metrics['auc_roc']:.3f} "
                f"logit_μ={val_metrics['logit_mean']:+.2f} logit_σ={val_metrics['logit_std']:.2f}"
                + (f" | {gpu_info}" if gpu_info else ""),
                flush=True,
            )

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_metrics = val_metrics.copy()
                patience_counter = 0
                state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state, ckpt_dir / f"fold_{fold}_best.pt")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch + 1}", flush=True)
                    break

        print(f"\n  Fold {fold+1} best: auc={best_metrics.get('auc_roc', 0):.3f} "
              f"f1={best_metrics.get('f1', 0):.3f} acc={best_metrics.get('accuracy', 0):.3f}", flush=True)
        fold_metrics.append(best_metrics)

        # Free GPU memory before the next fold's model is allocated
        del model, optimizer, scheduler, scaler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    monitor.stop()

    # Aggregate
    print("\n" + "="*60)
    print("Cross-Validation Results")
    print("="*60)
    for metric in ["accuracy", "f1", "auc_roc"]:
        values = [m.get(metric, 0) for m in fold_metrics]
        print(f"  {metric:12s}: {np.mean(values):.3f} ± {np.std(values):.3f}")

    return fold_metrics


def main():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"DataParallel: {'yes' if n_gpus > 1 else 'no (single GPU)'}")
    else:
        print("CUDA not available, using CPU")
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (0=safe default; 4 after extract_frames.py)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (effective_batch = batch_size × steps)")
    parser.add_argument("--n_frames", type=int, default=None,
                        help="Frames sampled per video (default 16; use 64 for max quality)")
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
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.grad_accum is not None:
        config.grad_accum_steps = args.grad_accum
    if args.n_frames is not None:
        config.n_frames = args.n_frames

    print(f"Device:          {config.device}")
    print(f"Manifest:        {config.manifest_csv}")
    print(f"Features:        {config.feature_dir}")
    print(f"Batch size:      {config.batch_size}  "
          f"(effective {config.batch_size * config.grad_accum_steps} "
          f"with grad_accum={config.grad_accum_steps})")
    print(f"num_workers:     {config.num_workers}")
    print(f"n_frames:        {config.n_frames}")
    print(f"cnn_chunk_size:  {config.cnn_chunk_size}")

    run_cross_validation(config)


if __name__ == "__main__":
    main()
