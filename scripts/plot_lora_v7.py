import matplotlib.pyplot as plt

# Data extracted from training log
data = [
    (1, 1.097, 2.065),
    (5, 1.065, 1.394),
    (10, 0.9943, 1.016),
    (15, 0.9639, 0.5781),
    (20, 0.9509, 0.5412),
    (25, 0.9193, 0.3288),
    (30, 0.925, 0.4024),
    (35, 0.906, 0.2418),
    (40, 0.898, 0.2533),
    (45, 0.8927, 0.4072),
    (50, 0.8904, 0.4483),
    (55, 0.8889, 0.5907),
    (60, 0.895, 0.3307),
    (65, 0.8804, 0.3737),
    (70, 0.8881, 0.4725),
    (75, 0.9103, 0.5442),
    (80, 0.9198, 0.4013),
    (85, 0.8996, 0.42),
    (90, 0.9258, 0.3748),
    (95, 0.9003, 0.3037),
    (100, 0.891, 0.4873),
    (105, 0.8877, 0.2901),
    (110, 0.8882, 0.233),
    (115, 0.8846, 0.2167),
    (120, 0.8795, 0.1692),
    (125, 0.8831, 0.3152),
    (130, 0.8774, 3.39),
    (135, 1.034, 0.4894),
    (140, 0.9294, 0.9015),
    (145, 0.9042, 0.3026),
    (150, 0.905, 0.3495),
    (155, 0.8969, 0.3707),
    (160, 0.9217, 0.3811),
    (165, 0.8935, 0.2002),
    (170, 3.225, 64.28),
    (175, 1.952, 37.35),
    (180, 3.446, 54.15),
    (185, 8.458, 34.26),
    (190, 6.917, 23.62),
    (195, 6.111, 7.876),
    (200, 6.0, 3.608),
    (205, 5.691, 3.885),
    (210, 5.433, 1.869),
    (215, 5.272, 1.696),
    (220, 5.196, 2.153),
]

steps = [d[0] for d in data]
losses = [d[1] for d in data]
grad_norms = [d[2] for d in data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(steps, losses, 'b-o', markersize=3)
ax1.axhline(y=0.88, color='g', linestyle='--', alpha=0.5, label='min loss before explosion')
ax1.axvline(x=170, color='r', linestyle='--', alpha=0.5, label='explosion @ step 170')
ax1.set_ylabel('Loss')
ax1.set_title('LoRA v7 (lr=1e-3, α=256) — Loss Explosion')
ax1.legend()
ax1.set_ylim(0, 10)
ax1.grid(True, alpha=0.3)

ax2.plot(steps, grad_norms, 'r-o', markersize=3)
ax2.axvline(x=130, color='orange', linestyle='--', alpha=0.5, label='grad spike @ step 130')
ax2.axvline(x=170, color='r', linestyle='--', alpha=0.5, label='explosion @ step 170')
ax2.set_ylabel('Grad Norm')
ax2.set_xlabel('Step')
ax2.legend()
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sft_lora_v7_explosion.png', dpi=150, bbox_inches='tight')
print("Saved: sft_lora_v7_explosion.png")
