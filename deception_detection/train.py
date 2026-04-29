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
import json
import os
import random
import threading
from dataclasses import asdict
from datetime import datetime
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
from .models.model_utils import print_module_summary


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

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        f_tensor = batch["frames"]
        print(f"\r[{step + 1} / {len(loader)}] [{f_tensor.max().item():.2f}] Working...", end="", flush=True)
        
        #if f_tensor.std() < 0.1:
        #    print("  WARNING: Low variance in frames. Data might be corrupted or loading incorrectly.")
        

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
            optimizer.zero_grad(set_to_none = True)

        # Track unscaled loss (multiply back)
        total_loss += loss.item() * grad_accum_steps * logits.size(0)
        preds = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().tolist())

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])

    # At the very end of train_one_epoch and evaluate
    #del all_preds, all_labels
    import gc
    gc.collect()
    
    return avg_loss, acc


@torch.inference_mode()
def evaluate(model, loader, pos_weight, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
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

    # At the very end of train_one_epoch and evaluate
    #del all_preds, all_labels
    import gc
    gc.collect()

    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, binary_preds),
        "f1": f1,
        "auc_roc": auc,
        "logit_mean": float(np.mean(all_preds)),
        "logit_std": float(np.std(all_preds)),
    }


def _capture_rng_states() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_states(state: dict) -> None:
    if state is None:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print(f"  [warn] could not restore RNG state: {e}", flush=True)


def _save_full_checkpoint(
    path: Path,
    *,
    fold: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    best_val_auc: float,
    best_metrics: dict,
    patience_counter: int,
    fold_metrics: list,
    config: ModelConfig,
    run_id: str,
) -> None:
    """Save full training state for resumability (model + optimizer + sched + scaler + bookkeeping)."""
    state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    payload = {
        "schema_version": 1,
        "fold": fold,
        "epoch": epoch,
        "run_id": run_id,
        "model_state": state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_auc": float(best_val_auc),
        "best_metrics": dict(best_metrics) if best_metrics else {},
        "patience_counter": int(patience_counter),
        "fold_metrics": list(fold_metrics),
        "config": asdict(config),
        "rng_states": _capture_rng_states(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)                                   # atomic on POSIX


def _load_full_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    required = {"fold", "epoch", "model_state", "optimizer_state", "config"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint at {path} missing required keys: {missing}")
    return payload


def run_cross_validation(config: ModelConfig):
    device = torch.device(config.device)
    root = Path(__file__).resolve().parents[1]
    manifest = root / config.manifest_csv
    feature_dir = root / config.feature_dir

    # ------------------------------------------------------------------
    # Resume bookkeeping
    # ------------------------------------------------------------------
    resume_payload: dict | None = None
    if config.resume_from:
        resume_path = Path(config.resume_from)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_payload = _load_full_checkpoint(resume_path)
        run_id = resume_payload["run_id"]
        ckpt_dir = resume_path.parent
        print(f"Resuming from:   {resume_path}", flush=True)
        print(f"  Saved at fold={resume_payload['fold']} epoch={resume_payload['epoch']}", flush=True)
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = root / config.checkpoint_dir / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir:         {ckpt_dir}", flush=True)

    # A) Persist run configuration (overwrite on resume so post-resume overrides are recorded)
    config_path = ckpt_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    # B) Per-fold metrics JSON (rewritten after each fold so partial runs are saved)
    metrics_path = ckpt_dir / "metrics.json"
    last_ckpt_path = ckpt_dir / "last.pt"

    # Load full dataset for CV splitting
    full_dataset = DeceptionDataset(
        str(manifest), str(feature_dir),
        augment=False,
        max_frames=config.max_frames,
        legacy_n_frames=config.legacy_n_frames,
    )
    labels = full_dataset.get_labels()
    indices = list(range(len(full_dataset)))

    if config.n_folds == 1:
        # Smoke-test mode: single 80/20 stratified split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.seed)
        splits = list(sss.split(indices, labels))
    else:
        skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
        splits = list(skf.split(indices, labels))

    # Record which sample IDs went into train vs val for each fold
    sample_ids = [sid for sid, _ in full_dataset.samples]
    splits_record = [
        {
            "fold": i,
            "train": [
                {"sample_id": sample_ids[j], "label": int(labels[j])}
                for j in train_idx.tolist()
            ],
            "val": [
                {"sample_id": sample_ids[j], "label": int(labels[j])}
                for j in val_idx.tolist()
            ],
        }
        for i, (train_idx, val_idx) in enumerate(splits)
    ]
    with (ckpt_dir / "splits.json").open("w") as f:
        json.dump(splits_record, f, indent=2)

    # Restore prior fold metrics if resuming.
    fold_metrics = list(resume_payload["fold_metrics"]) if resume_payload else []
    resume_fold = resume_payload["fold"] if resume_payload else -1
    resume_epoch = resume_payload["epoch"] if resume_payload else -1
    if resume_payload is not None:
        _restore_rng_states(resume_payload.get("rng_states"))

    monitor = _GPUMonitor(interval=2)
    monitor.start()

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Skip folds that were already completed before the resume checkpoint.
        if fold < resume_fold:
            print(f"\n[resume] Skipping fold {fold + 1}/{len(splits)} — already completed.", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Fold {fold + 1}/{len(splits)}", flush=True)
        print(f"{'='*60}", flush=True)

        # Fold-specific datasets
        full_dataset = DeceptionDataset(
            str(manifest), str(feature_dir),
            augment=True,
            max_frames=config.max_frames,
            legacy_n_frames=config.legacy_n_frames,
        )

        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)


        # train_dataset = DeceptionDataset(
        #     str(manifest), str(feature_dir),
        #     augment=True,
        #     max_frames=config.max_frames,
        #     legacy_n_frames=config.legacy_n_frames,
        # )
        # train_dataset.samples = [train_dataset.samples[i] for i in train_idx]
        # val_dataset = DeceptionDataset(
        #     str(manifest), str(feature_dir),
        #     augment=False,
        #     max_frames=config.max_frames,
        #     legacy_n_frames=config.legacy_n_frames,
        # )
        # val_dataset.samples = [val_dataset.samples[i] for i in val_idx]

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
            pin_memory=False,
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
            pin_memory=False,
            persistent_workers=config.num_workers > 0,
            **extra,
        )

        # Model, optimizer, scheduler, AMP scaler
        model = FusionModel(config).to(device)
        if torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs (DataParallel)", flush=True)
            model = torch.nn.DataParallel(model)
        if hasattr(torch, "compile"):
            #model = torch.compile(model, mode="reduce-overhead")
            model = model.to(device)

        # Detailed module-level frozen/trainable breakdown
        underlying = model.module if isinstance(model, torch.nn.DataParallel) else model
        print_module_summary(underlying)

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
        best_metrics: dict = {}
        patience_counter = 0
        start_epoch = 0

        # Restore mid-fold state on resume.
        if resume_payload is not None and fold == resume_fold:
            target = model.module if isinstance(model, torch.nn.DataParallel) else model
            target.load_state_dict(resume_payload["model_state"])
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            if resume_payload.get("scheduler_state") is not None:
                scheduler.load_state_dict(resume_payload["scheduler_state"])
            if resume_payload.get("scaler_state") is not None:
                scaler.load_state_dict(resume_payload["scaler_state"])
            best_val_auc = float(resume_payload.get("best_val_auc", 0.0))
            best_metrics = dict(resume_payload.get("best_metrics") or {})
            patience_counter = int(resume_payload.get("patience_counter", 0))
            start_epoch = int(resume_payload["epoch"]) + 1
            print(f"  [resume] continuing from epoch {start_epoch + 1}/{config.max_epochs}, "
                  f"best_auc={best_val_auc:.3f}, patience={patience_counter}", flush=True)
            # Only consume the resume payload once.
            resume_payload = None

        for epoch in range(start_epoch, config.max_epochs):
            monitor.reset_epoch()
            print(f"=== Epoch {epoch+1}/{config.max_epochs} ===", flush=True)
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                pos_weight, device, config.grad_clip,
                grad_accum_steps=config.grad_accum_steps,
                scaler=scaler,
                label_smoothing=config.label_smoothing,
            )
            val_metrics = evaluate(model, val_loader, pos_weight, device)
            gpu_info = monitor.epoch_summary()

            cur_lr = scheduler.get_last_lr()[0]
            print(
                f"\n    === Epoch {epoch+1:3d} Summary ===" +
                f"\n    lr={cur_lr:.2e} ".ljust(24) +
                f"train_loss={train_loss:.4f} ".ljust(20) +
                f"train_acc={train_acc:.3f}" +
                f"\n    val_loss={val_metrics['loss']:.4f} ".ljust(24) +
                f"val_acc={val_metrics['accuracy']:.3f} ".ljust(20) +
                f"val_f1={val_metrics['f1']:.3f} ".ljust(20) +
                f"val_auc={val_metrics['auc_roc']:.3f} " +
                f"\n    logit_μ={val_metrics['logit_mean']:+.2f} ".ljust(24) +
                f"logit_σ={val_metrics['logit_std']:.2f} ".ljust(20) +
                f"{(f"{gpu_info}" if gpu_info else "")}",
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

            # Always save the latest full training state for resumability — written after
            # the best-checkpoint update so ``last.pt`` reflects the current best_val_auc.
            _save_full_checkpoint(
                last_ckpt_path,
                fold=fold, epoch=epoch,
                model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                best_val_auc=best_val_auc, best_metrics=best_metrics,
                patience_counter=patience_counter,
                fold_metrics=fold_metrics,
                config=config, run_id=run_id,
            )

            if patience_counter >= config.patience:
                print(f"  Early stopping at epoch {epoch + 1}", flush=True)
                break

        print(f"\n  Fold {fold+1} best: auc={best_metrics.get('auc_roc', 0):.3f} "
              f"f1={best_metrics.get('f1', 0):.3f} acc={best_metrics.get('accuracy', 0):.3f}", flush=True)
        fold_metrics.append(best_metrics)

        # Persist metrics incrementally — one object per fold
        with metrics_path.open("w") as f:
            json.dump(
                [{"fold": i, **m} for i, m in enumerate(fold_metrics)],
                f,
                indent=2,
            )

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
    
    
    parser = argparse.ArgumentParser(
        description="Train the bidirectional UT-Adapter fusion model with k-fold CV.",
    )
    # I/O
    parser.add_argument("--config", default=None, help="Path to JSON config; CLI flags override.")
    parser.add_argument("--resume", default=None, help="Path to last.pt to resume training from.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--device", default=None)

    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (0=safe default; 4 after extract_frames.py)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (effective_batch = batch_size × steps)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Cap on frames per clip (None = use all).")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate (OneCycleLR max_lr).")
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--label_smoothing", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # Adapters / fusion
    parser.add_argument("--use_ut_adapters", type=lambda v: v.lower() in ("1","true","yes","y","t"), default=None)
    parser.add_argument("--ut_adapter_dim", type=int, default=None)
    parser.add_argument("--ut_conv_kernel", type=int, default=None)
    parser.add_argument("--audio_fusion_layers", default=None,
                        help="Comma-separated 1-indexed Wav2Vec2 layer indices, e.g. 4,8,12.")
    parser.add_argument("--visual_fusion_layers", default=None,
                        help="Comma-separated 1-indexed temporal-layer indices, e.g. 1,2,4.")
    parser.add_argument("--fusion_aggregator", default=None, choices=["sum", "weighted_sum"])
    parser.add_argument("--fusion_n_heads", type=int, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=None)
    parser.add_argument("--wav2vec2_unfreeze_last_n", type=int, default=None,
                        help="Only used when --use_ut_adapters=False.")
    parser.add_argument("--vit_n_layers", type=int, default=None)
    parser.add_argument("--vit_n_heads", type=int, default=None)
    parser.add_argument("--freeze_visual_backbone", type=lambda v: v.lower() in ("1","true","yes","y","t"), default=None)
    parser.add_argument("--visual_backbone", default=None, choices=["clip", "arcface"],
                        help="Pretrained visual backbone (frozen).")
    parser.add_argument("--visual_backbone_model", default=None,
                        help="HuggingFace model id for --visual_backbone=clip (e.g. openai/clip-vit-large-patch14).")
    args = parser.parse_args()

    # Build config: JSON file (if any) -> CLI overrides.
    if args.config:
        config = ModelConfig.from_json(args.config)
    else:
        config = ModelConfig()

    def _parse_int_csv(s: str) -> tuple:
        return tuple(int(x) for x in s.split(",") if x.strip())

    overrides = {
        "manifest_csv": args.manifest,
        "feature_dir": args.feature_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_epochs": args.epochs,
        "n_folds": args.folds,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "grad_accum_steps": args.grad_accum,
        "max_frames": args.max_frames,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "dropout": args.dropout,
        "use_ut_adapters": args.use_ut_adapters,
        "ut_adapter_dim": args.ut_adapter_dim,
        "ut_conv_kernel": args.ut_conv_kernel,
        "audio_fusion_layers": _parse_int_csv(args.audio_fusion_layers) if args.audio_fusion_layers else None,
        "visual_fusion_layers": _parse_int_csv(args.visual_fusion_layers) if args.visual_fusion_layers else None,
        "fusion_aggregator": args.fusion_aggregator,
        "fusion_n_heads": args.fusion_n_heads,
        "fusion_dropout": args.fusion_dropout,
        "wav2vec2_unfreeze_last_n": args.wav2vec2_unfreeze_last_n,
        "vit_n_layers": args.vit_n_layers,
        "vit_n_heads": args.vit_n_heads,
        "freeze_visual_backbone": args.freeze_visual_backbone,
        "visual_backbone": args.visual_backbone,
        "visual_backbone_model": args.visual_backbone_model,
        "resume_from": args.resume,
    }
    for k, v in overrides.items():
        if v is not None:
            setattr(config, k, v)
    # Re-run validation after CLI overrides.
    config.__post_init__()

    print(f"Device:          {config.device}")
    print(f"Manifest:        {config.manifest_csv}")
    print(f"Features:        {config.feature_dir}")
    print(f"Batch size:      {config.batch_size}  "
          f"(effective {config.batch_size * config.grad_accum_steps} "
          f"with grad_accum={config.grad_accum_steps})")
    print(f"num_workers:     {config.num_workers}")
    print(f"max_frames:      {config.max_frames}")
    print(f"cnn_chunk_size:  {config.cnn_chunk_size}")
    print(f"Visual backbone: {config.visual_backbone}  ({config.visual_backbone_model if config.visual_backbone == 'clip' else 'vggface2'})")
    print(f"UT-Adapters:     {config.use_ut_adapters}  (bottleneck={config.ut_adapter_dim}, k={config.ut_conv_kernel})")
    print(f"Audio fusion:    {config.audio_fusion_layers}  (Wav2Vec2 layer indices)")
    print(f"Visual fusion:   {config.visual_fusion_layers}  (temporal-layer indices)")
    print(f"Fusion agg:      {config.fusion_aggregator}  (n_heads={config.fusion_n_heads}, dropout={config.fusion_dropout})")
    if config.resume_from:
        print(f"Resume from:     {config.resume_from}")

    run_cross_validation(config)


if __name__ == "__main__":
    main()
