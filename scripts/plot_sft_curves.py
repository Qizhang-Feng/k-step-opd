import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def load_log(path):
    steps, losses, grad_norms = [], [], []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
                # ms-swift format: "global_step/max_steps" like "5/487"
                step_str = d.get("global_step/max_steps") or d.get("global_step")
                loss = d.get("loss")
                gn = d.get("grad_norm")
                if step_str is not None and loss is not None:
                    if isinstance(step_str, str) and "/" in step_str:
                        step = int(step_str.split("/")[0])
                    else:
                        step = int(step_str)
                    steps.append(step)
                    losses.append(float(loss))
                    grad_norms.append(float(gn) if gn is not None else float('nan'))
            except:
                pass
    return steps, losses, grad_norms

# Load data
steps_4b, losses_4b, gn_4b = load_log("/tmp/4b_logging.jsonl")
steps_8b, losses_8b, gn_8b = load_log("/tmp/8b_logging.jsonl")
steps_4b_2ep, losses_4b_2ep, gn_4b_2ep = load_log("/tmp/4b_2ep_logging.jsonl")
steps_4b_v2, losses_4b_v2, gn_4b_v2 = load_log("/tmp/4b_v2_logging.jsonl")

print(f"4B 1ep (179K):  {len(steps_4b)} steps, final loss={losses_4b[-1]:.4f}")
print(f"8B 1ep (179K):  {len(steps_8b)} steps, final loss={losses_8b[-1]:.4f}")
print(f"4B 2ep (179K):  {len(steps_4b_2ep)} steps, current loss={losses_4b_2ep[-1]:.4f}")
print(f"4B v2 (79K×3ep):{len(steps_4b_v2)} steps, final loss={losses_4b_v2[-1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('4B/8B Full FT SFT Training Curves', fontsize=14, fontweight='bold')

# All 4B runs comparison
ax = axes[0][0]
ax.plot(steps_4b_v2, losses_4b_v2, 'purple', linewidth=1.5, label='4B v2 (79K filtered, 3ep) → AIME 50%')
ax.plot(steps_4b, losses_4b, 'b-', linewidth=1.5, label='4B 1ep (179K, 1ep) → AIME 0%')
ax.plot(steps_4b_2ep, losses_4b_2ep, 'g-', linewidth=1.5, label=f'4B 2ep (179K, running, {steps_4b_2ep[-1]}/974)')
ax.plot(steps_8b, losses_8b, 'r-', linewidth=1.5, label='8B 1ep (179K, 1ep) → AIME 63.3%')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title('All Runs — Loss Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 4B v2 (successful baseline)
ax = axes[0][1]
ax.plot(steps_4b_v2, losses_4b_v2, 'purple', linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'4B v2 (79K×3ep) — Loss (final={losses_4b_v2[-1]:.4f}) ✅ AIME 50%')
ax.grid(True, alpha=0.3)

# 4B 2ep (running)
ax = axes[1][0]
ax.plot(steps_4b_2ep, losses_4b_2ep, 'g-', linewidth=1.5)
ax.axvline(x=487, color='gray', linestyle='--', alpha=0.5, label='epoch 1 end (~487)')
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'4B 2ep (179K) — Loss (current={losses_4b_2ep[-1]:.4f}, step {steps_4b_2ep[-1]}/974)')
ax.legend()
ax.grid(True, alpha=0.3)

# 8B 1ep
ax = axes[1][1]
ax.plot(steps_8b, losses_8b, 'r-', linewidth=1.5)
ax.set_xlabel('Step')
ax.set_ylabel('Loss')
ax.set_title(f'8B 1ep (179K) — Loss (final={losses_8b[-1]:.4f}) ✅ AIME 63.3%')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sft_full_4b_8b_179k_curves.png', dpi=150, bbox_inches='tight')
print("Saved: sft_full_4b_8b_179k_curves.png")

# Summary stats
print(f"\n4B v2 (79K×3ep): init={losses_4b_v2[0]:.4f} → final={losses_4b_v2[-1]:.4f} (↓{(losses_4b_v2[0]-losses_4b_v2[-1])/losses_4b_v2[0]*100:.1f}%) → AIME 50%")
print(f"4B 1ep (179K):   init={losses_4b[0]:.4f} → final={losses_4b[-1]:.4f} (↓{(losses_4b[0]-losses_4b[-1])/losses_4b[0]*100:.1f}%) → AIME 0%")
print(f"8B 1ep (179K):   init={losses_8b[0]:.4f} → final={losses_8b[-1]:.4f} (↓{(losses_8b[0]-losses_8b[-1])/losses_8b[0]*100:.1f}%) → AIME 63.3%")
print(f"4B 2ep (179K):   init={losses_4b_2ep[0]:.4f} → current={losses_4b_2ep[-1]:.4f} (step {steps_4b_2ep[-1]}/974, running)")
