"""Pretty-print a frozen/trainable breakdown of a model's submodules.

Used at the start of each training fold to make it crystal-clear which parts of
the network are receiving gradient updates (e.g. UT-Adapters + AdapterNorms +
fusion + head) versus which are kept frozen (Wav2Vec2 backbone).
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch.nn as nn


_LIST_INDEX = re.compile(r"\.\d+\.")


def _key(param_name: str) -> str:
    """Collapse list indices: ``layers.0.attn.q_proj.weight`` -> ``layers[*].attn.q_proj.weight``."""
    return _LIST_INDEX.sub("[*].", param_name)


def _module_path(param_name: str) -> str:
    """Return the parent module path of a parameter.

    For a parameter belonging to a sub-module (e.g. ``foo.bar.weight``), this is
    the parent module (``foo.bar``). For a bare ``nn.Parameter`` registered
    directly on a module (e.g. ``foo.pos_embedding``), the strip would collapse
    onto the parent module name and lose the parameter name; we instead keep
    the full name so it's distinguishable from sub-module rows.
    """
    parent, _, leaf = param_name.rpartition(".")
    # nn.Module sub-parameter conventions; everything else (custom nn.Parameter
    # names like ``pos_embedding``) we keep verbatim so the row label includes it.
    if leaf in {"weight", "bias", "in_proj_weight", "in_proj_bias",
                "out_proj_weight", "out_proj_bias", "running_mean",
                "running_var", "num_batches_tracked"}:
        return parent
    return param_name


def _format_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:6.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:6.2f}K"
    return f"{n:6d}  "


def summarize_modules(model: nn.Module, max_rows_per_top: int = 200) -> str:
    """Return a multiline string with per-module trainable/frozen counts.

    Modules are grouped by collapsed key (list indices replaced with [*]) so a
    Wav2Vec2 stack of 12 identical layers shows up once with a ``×12`` multiplier.
    """
    # Collect per-collapsed-module-path counts.
    rows: "OrderedDict[str, dict]" = OrderedDict()
    for name, p in model.named_parameters():
        mod_path = _module_path(name)
        key = _key(mod_path)
        # Track replication multiplier: how many distinct list indices map to this collapsed key?
        instance_id = mod_path  # full uncollapsed path
        bucket = rows.setdefault(
            key,
            {"trainable": 0, "total": 0, "instances": set(), "any_grad": False, "all_grad": True},
        )
        bucket["total"] += p.numel()
        if p.requires_grad:
            bucket["trainable"] += p.numel()
            bucket["any_grad"] = True
        else:
            bucket["all_grad"] = False
        bucket["instances"].add(instance_id)

    # Group rows by top-level submodule (e.g. ``audio_model``, ``visual_model``, ``fusion``, ``head``).
    grouped: "OrderedDict[str, list[tuple[str, dict]]]" = OrderedDict()
    for key, info in rows.items():
        top = key.split(".", 1)[0]
        grouped.setdefault(top, []).append((key, info))

    lines: list[str] = []
    sep = "=" * 92
    lines.append(sep)
    lines.append(f" {'Module':<70s} {'Status':<8s} {'Trainable / Total params':>14s}")
    lines.append(sep)

    grand_tr, grand_tot = 0, 0
    for top, entries in grouped.items():
        # Alphabetic sort on the collapsed key keeps each submodule's rows
        # contiguous (e.g. all ``spatial_encoder.*`` together; all
        # ``temporal_layers[*].*`` together) instead of interleaving by depth.
        entries.sort(key=lambda kv: kv[0])
        top_tr = sum(e[1]["trainable"] for e in entries)
        top_tot = sum(e[1]["total"] for e in entries)
        grand_tr += top_tr
        grand_tot += top_tot

        status = "TRAIN" if top_tr == top_tot else ("FROZEN" if top_tr == 0 else "MIXED")
        lines.append(
            f"[{top}]".ljust(71)
            + f" {status:<8s} {_format_n(top_tr)} / {_format_n(top_tot)}"
        )
        for i, (key, info) in enumerate(entries):
            if i >= max_rows_per_top:
                lines.append(f"   ... ({len(entries) - i} more rows truncated)")
                break
            n_inst = len(info["instances"])
            mult = f" ×{n_inst}" if n_inst > 1 else ""
            sub_status = (
                "TRAIN" if info["all_grad"]
                else "FROZEN" if not info["any_grad"]
                else "MIXED"
            )
            lines.append(
                f"  {key + mult:<69s} {sub_status:<8s} "
                f"{_format_n(info['trainable'])} / {_format_n(info['total'])}"
            )
        lines.append("")

    lines.append(sep)
    pct = 100 * grand_tr / max(grand_tot, 1)
    lines.append(
        f" TOTAL{'':<65s} {'':8s} "
        f"{_format_n(grand_tr)} / {_format_n(grand_tot)}  ({pct:.2f}% trainable)"
    )
    lines.append(sep)
    return "\n".join(lines)


def print_module_summary(model: nn.Module, **kwargs) -> None:
    print(summarize_modules(model, **kwargs), flush=True)
