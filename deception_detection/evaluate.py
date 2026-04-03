"""
Standalone evaluation script for a saved model checkpoint.

Usage:
    python -m deception_detection.evaluate \
        --checkpoint checkpoints/fold_0_best.pt \
        --manifest Data/manifest_dolos.csv \
        --feature_dir features
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from .config import ModelConfig
from .data.dataset import DeceptionDataset
from .data.collate import collate_fn
from .models.full_model import MultimodalDeceptionModel
from .train import evaluate as _evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--manifest", default="Data/manifest_dolos.csv")
    parser.add_argument("--feature_dir", default="features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config = ModelConfig()
    config.device = args.device

    device = torch.device(args.device)

    dataset = DeceptionDataset(
        str(root / args.manifest),
        str(root / args.feature_dir),
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    model = MultimodalDeceptionModel(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # pos_weight = 1 for evaluation (not used in metric calculation, only loss)
    pos_weight = torch.tensor([1.0])
    metrics = _evaluate(model, loader, pos_weight, device)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    main()
