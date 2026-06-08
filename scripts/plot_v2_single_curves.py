#!/usr/bin/env python3
"""Plot loss / lr / grad_norm curves for v2 single-node training and overlay v2 multi-node for comparison."""
import json
import os
import sys
import matplotlib.pyplot as plt

LOGS = [
    ("v2-single (8 GPU)",  "/root/.cache/huggingface/sft-qwen3-4b-full-v2-single/v0-20260523-000055/logging.jsonl"),
    ("v2-multi (16 GPU)", "/root/.cache/huggingface/sft-qwen3-4b-full-teacher-v2/v9-20260515-233350/logging.jsonl"),
]
OUT = "/workspace/k-step-opd/sft_v2_single_vs_multi.png"

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "loss" in d:
                # parse "global_step/max_steps": "541/759" -> int 541
                step = None
                if "global_step" in d:
                    step = d["global_step"]
                elif "global_step/max_steps" in d:
                    try:
                        step = int(d["global_step/max_steps"].split("/")[0])
                    except Exception:
                        pass
                if step is not None:
                    d["global_step"] = step
                    rows.append(d)
    return rows

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for label, path in LOGS:
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    rows = load(path)
    if not rows:
        print(f"EMPTY: {path}")
        continue
    steps = [r["global_step"] for r in rows]
    loss = [r["loss"] for r in rows]
    grad = [r.get("grad_norm", 0) for r in rows]
    lr = [r.get("learning_rate", 0) for r in rows]

    axes[0].plot(steps, loss, label=label, alpha=0.85)
    axes[1].plot(steps, grad, label=label, alpha=0.85)
    axes[2].plot(steps, lr, label=label, alpha=0.85)

for ax, title, ylabel in [
    (axes[0], "Train Loss", "loss"),
    (axes[1], "Grad Norm", "grad_norm"),
    (axes[2], "Learning Rate", "lr"),
]:
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()

plt.suptitle("Qwen3-4B-Base Full FT v2 (79K filtered × 3ep) — Single-node vs Multi-node")
plt.tight_layout()
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"Saved: {OUT}")
