"""
Plot all SFT experiments on ONE chart for direct comparison.

Naming:
  4B/8B: model size
  data composition:
    79K-A   = teacher@temp=0.6, top_p=0.95 only        (style A)
    73K-B   = teacher@temp=0.7, top_p=0.9 only         (style B)
    179K-AB = mixed A + B (unfiltered)
    152K-AB = mixed A + B (think-filtered)
  results: → X%  (AIME-2024 pass@1 at the noted ckpt)
"""
import json
import matplotlib.pyplot as plt


def load_log(path):
    steps, losses = [], []
    try:
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    s = d.get("global_step/max_steps") or d.get("global_step")
                    l = d.get("loss")
                    if s is not None and l is not None:
                        if isinstance(s, str) and "/" in s:
                            step = int(s.split("/")[0])
                        else:
                            step = int(s)
                        steps.append(step)
                        losses.append(float(l))
                except Exception:
                    pass
    except Exception:
        pass
    return steps, losses


# (label, file, color, linestyle, linewidth)
EXPERIMENTS = [
    # ---- 4B Full FT, single-style data (works) ----
    ("4B FullFT 79K-A  ×3ep, 16gpu  → 60% (n=1) / 49% (n=16)",  "/tmp/4b_v2_logging.jsonl",         "purple",   "-",  2.0),
    ("4B FullFT 79K-A  ×3ep,  8gpu  → 47% (n=1)",               "/tmp/4b_v2_single_logging.jsonl",  "indigo",   "--", 1.5),
    ("4B FullFT 73K-B  ×3ep,  8gpu  → 53% (n=1)",               "/tmp/4b_73k_new_logging.jsonl",    "teal",     "-",  2.0),
    # ---- 4B Full FT, mixed-style data ----
    ("4B FullFT 179K-AB ×1ep        →  0%",                     "/tmp/4b_logging.jsonl",            "lightblue","-",  1.5),
    ("4B FullFT 179K-AB ×2ep        →  3%",                     "/tmp/4b_2ep_logging.jsonl",        "skyblue",  "-",  1.5),
    ("4B FullFT 179K-AB ×3ep        →  3%",                     "/tmp/4b_3ep_logging.jsonl",        "blue",     "-",  1.5),
    ("4B FullFT 152K-AB ×3ep filt   →  3%",                     "/tmp/4b_152k_3ep_logging.jsonl",   "magenta",  "-",  1.5),
    # NEW: same 152K-AB data but capped at 700 steps (≈ same budget as v2)
    ("4B FullFT 152K-AB ×1.44ep, 700 step  → ?",                "/tmp/4b_152k_700steps_logging.jsonl", "deeppink", "-",  2.5),
    # ---- 8B Full FT (capacity check, works on mix) ----
    ("8B FullFT 179K-AB ×1ep        → 63% (n=1)",               "/tmp/8b_logging.jsonl",            "red",      "-",  2.5),
    # ---- LoRA (all 4B fail; 8B 50%) ----
    ("4B LoRA v5  +lm_head, 100K OT3  →  0%",                   "/tmp/lora_v5_logging_real.jsonl",  "gray",     ":",  1.0),
    ("4B LoRA v7  lr=1e-3, EXPLODED",                           "/tmp/lora_v7_logging.jsonl",       "orange",   ":",  1.0),
    ("4B LoRA v8  no lm_head        →  0%",                     "/tmp/lora_v8_logging.jsonl",       "brown",    ":",  1.0),
    ("4B LoRA v9  Tinker recipe     →  0%",                     "/tmp/lora_v9_logging.jsonl",       "olive",    ":",  1.0),
    # ("8B LoRA    100K OT3           → 50%",                     "/tmp/lora_8b_logging.jsonl",       "darkcyan", ":",  1.0),  # log lost
]


fig, ax = plt.subplots(figsize=(14, 9))

for label, path, color, ls, lw in EXPERIMENTS:
    steps, losses = load_log(path)
    if steps:
        ax.plot(steps, losses, color=color, linestyle=ls, linewidth=lw, label=label, alpha=0.9)
    else:
        print(f"WARN: no data for {label} ({path})")

ax.set_title(
    "All SFT experiments (Qwen3 4B/8B Base on math)\n"
    "style A = teacher@temp=0.6,top_p=0.95   |   style B = teacher@temp=0.7,top_p=0.9",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("training step", fontsize=11)
ax.set_ylabel("loss", fontsize=11)
ax.set_ylim(0, 1.4)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper right", framealpha=0.95)

plt.tight_layout()
out = "sft_all_experiments_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Summary
print("\n=== Summary (final loss / steps) ===")
for label, path, _, _, _ in EXPERIMENTS:
    steps, losses = load_log(path)
    if steps:
        print(f"  {label:<55s}  steps={steps[-1]:>5d}  loss={losses[-1]:.3f}")
