# Naming convention — 4B OPD experiments

## Two runs to date

| Short name | Date | Hyperparam | Rollouts | Best AIME-2024 (n=16) |
|:----------:|:----:|---|:---:|:---:|
| **opd-4b-A** | 5/24 | conservative (hand-picked) | 300 | 54.4% (iter_299) |
| **opd-4b-B** | 5/26-27 | matches Lightning-OPD paper Table 6 | 600 | 55.8% (iter_299) |

## Hyperparameter detail

| param | opd-4b-A (conservative) | opd-4b-B (paper) |
|---|:---:|:---:|
| LR | 5e-7 | **2e-6** |
| LR schedule | constant | constant |
| max-response-len | 8192 | **4096** |
| temperature | 0.6 | **0.8** |
| top-p | 0.95 | **1.0** (disabled) |
| top-k | 20 | **-1** (disabled) |
| num-rollout | 300 | **600** |
| rollout-batch-size | 16 | **64** |
| global-batch-size | 64 | **256** |
| n-samples-per-prompt | 4 | 4 |
| opd-kl-coef | 1.0 | 1.0 |
| max-tokens-per-gpu | 8192 | 16384 |
| advantage-estimator | grpo (+ use-opd) | grpo (+ use-opd) |
| Student | sft-qwen3-4b-full-v2-ckpt700 | same |
| Teacher | Qwen3-8B (qzf-dev DP=4 ext) | same |

## Disambiguation note

> "Lightning OPD" in the paper = the **offline precomputed-teacher** method.
> Our **opd-4b-B** is *standard online* OPD that just uses paper's **hyperparameters**.
> We did NOT implement Lightning-OPD's offline teacher precomputation.

## File location mapping

### checkpoints (Megatron torch_dist) on p5-3

| New name (use this) | Old name (existing on disk) |
|---|---|
| `~/.cache/huggingface/opd-4b-A/` | `~/.cache/huggingface/opd-4b-v2-ckpt700-instant/` |
| `~/.cache/huggingface/opd-4b-B/` | `~/.cache/huggingface/opd-4b-lightning-recipe/` |

### HF-converted checkpoints (replicated to p5-2/p5-3/p5-4)

| New | Old |
|---|---|
| `opd-4b-A-iter99-hf` | `opd-4b-v2-ckpt700-instant-iter99-hf` |
| `opd-4b-A-iter199-hf` | `opd-4b-v2-ckpt700-instant-iter199-hf` |
| `opd-4b-A-iter299-hf` | `opd-4b-v2-ckpt700-instant-iter299-hf` |
| `opd-4b-B-iter99-hf` | `opd-4b-lightning-iter99-hf` |
| `opd-4b-B-iter299-hf` | `opd-4b-lightning-iter299-hf` |
| `opd-4b-B-iter599-hf` | `opd-4b-lightning-iter599-hf` |

### configs (local)

| New | Old |
|---|---|
| `configs/opd-4b-A.env` | `configs/opd-4b-v2-ckpt700-instant-extteacher.env` |
| `configs/opd-4b-B.env` | `configs/opd-4b-lightning-recipe.env` |

### eval results

eval_results_n16/ filenames keep their original suffixes (rerunning is wasteful), but tables/plots use new labels:

| Eval JSON suffix | label |
|---|---|
| `_v2-ckpt700-baseline` | baseline (SFT) |
| `_opd-iter{99,199,299}` | opd-4b-A iter_{99,199,299} |
| `_opd-lightning-iter{99,299,599}` | opd-4b-B iter_{99,299,599} |
| `_152k-700steps` | sft-152k-700steps (Greenland) |

## Future runs

- **opd-4b-C**: cumulative K=8 OPD, K-step reward-to-go
- **opd-4b-D**: opd-4b-B but with Qwen3-32B teacher (larger gap)
- **opd-4b-E**: opd-4b-B + `--include-verifiable-reward` (task reward + KL combined)

If you forget which is which, look here.
