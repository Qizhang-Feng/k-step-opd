import matplotlib.pyplot as plt

# v9 metrics (sampled every 50 steps)
data = [
    (1, 0.5375, 0.1019),
    (50, 0.3816, 0.01514),
    (100, 0.345, 0.017),
    (150, 0.3244, 0.02064),
    (200, 0.3198, 0.02292),
    (250, 0.3106, 0.01798),
    (300, 0.3012, 0.01897),
    (350, 0.2997, 0.02792),
    (400, 0.2987, 0.02278),
    (450, 0.2937, 0.0209),
    (500, 0.2885, 0.01981),
    (505, None, None),  # epoch 1 boundary ~505
    (550, 0.2853, 0.01751),
    (600, 0.283, 0.01757),
    (650, 0.2842, 0.01914),
    (700, 0.2781, 0.01676),
    (750, 0.2795, 0.01877),
    (800, 0.2816, 0.015),
    (850, 0.276, 0.01402),
    (900, 0.2761, 0.01415),
    (950, 0.2769, 0.01301),
    (1000, 0.2747, 0.01477),
]

# Filter out None entries
data = [(s, l, g) for s, l, g in data if l is not None]
steps = [d[0] for d in data]
losses = [d[1] for d in data]
grad_norms = [d[2] for d in data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Loss
ax1.plot(steps, losses, 'b-o', markersize=4)
ax1.axvline(x=505, color='gray', linestyle='--', alpha=0.5, label='epoch 1 boundary')
ax1.axhline(y=0.246, color='g', linestyle=':', alpha=0.7, label='Full FT v2 final loss (0.246)')
ax1.set_ylabel('Loss')
ax1.set_title('LoRA v9 (α=32, lr=3e-4, teacher data, no lm_head)')
ax1.legend()
ax1.set_ylim(0.2, 0.6)
ax1.grid(True, alpha=0.3)

# Grad norm
ax2.plot(steps, grad_norms, 'r-o', markersize=4)
ax2.axvline(x=505, color='gray', linestyle='--', alpha=0.5, label='epoch 1 boundary')
ax2.set_ylabel('Grad Norm')
ax2.set_xlabel('Step')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sft_lora_v9_curves.png', dpi=150, bbox_inches='tight')
print("Saved: sft_lora_v9_curves.png")
print()
print("Summary:")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print(f"  Min loss:     {min(losses):.4f}")
print(f"  Total steps:  1010")
print(f"  Total time:   8h 17m")
print(f"  Avg speed:    29.6s/step")
