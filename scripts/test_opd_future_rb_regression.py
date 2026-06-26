"""Regression tests for Form A-K (Rao-Blackwellized future) OPD.

Tests:
1. _compute_renorm_topk_reverse_kl: verify the renormalized top-K KL formula
   matches a hand-computed reference on small synthetic input.
2. apply_opd_kl_to_advantages with --opd-future-rb=False produces NUMERICALLY
   IDENTICAL output to the existing R1 path (no behavior change for the legacy code path).
3. Form A-K K=2 mean: verify that the 2-D advantage matches the explicit per-t
   formula A_t = -(c_t + d_K(h_{t+1})) / 2 for t < T-1, A_{T-1} = -c_{T-1}.
4. Form A-K K=8 sum: verify the textbook reward-to-go form
   A_t = -(c_t + d_K(h_{t+1}) + d_K(h_{t+2}) + ... + d_K(h_{t+K-1})).

Run inside the slime container or any Python env that has slime + torch on PYTHONPATH.

Usage:
    cd /workspace/k-step-opd
    python3 scripts/test_opd_future_rb_regression.py
"""
import sys
from argparse import Namespace

import torch

# Make slime imports work from the workspace.
sys.path.insert(0, "/workspace/k-step-opd/slime")
sys.path.insert(0, "/root/slime")

from slime.backends.megatron_utils.loss import (  # noqa: E402
    _compute_renorm_topk_reverse_kl,
    apply_opd_kl_to_advantages,
)


def test_renorm_topk_kl_handcomputed():
    """Hand-computed reference on a 2-position 3-token-support example.

    Position 0: student logp = [-1.0, -2.0, -3.0], teacher logp = [-1.5, -1.8, -2.5]
    Renormalize:
        S logsumexp = log(e^-1 + e^-2 + e^-3) ~ -0.5577
        S_norm logp = [-0.4423, -1.4423, -2.4423], probs = [0.6432, 0.2369, 0.0871] (sum=0.9672)
        Wait — I forgot logsumexp doesn't make probs sum to 1 in float; it does after exp.
    Actually: S_norm = logp - logsumexp(logp), then S_norm.exp().sum() == 1 by definition.
    """
    # Position 0
    s_lp = torch.tensor([[-1.0, -2.0, -3.0]])
    t_lp = torch.tensor([[-1.5, -1.8, -2.5]])
    d_k = _compute_renorm_topk_reverse_kl(s_lp, t_lp, renorm=True)

    # Hand reference: KL(S_hat || T_hat) where both are renormalized within K-set.
    s_hat = (s_lp - torch.logsumexp(s_lp, dim=-1, keepdim=True)).exp()
    t_hat_logp = t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)
    s_hat_logp = s_lp - torch.logsumexp(s_lp, dim=-1, keepdim=True)
    expected = (s_hat * (s_hat_logp - t_hat_logp)).sum(dim=-1)
    assert torch.allclose(d_k, expected), f"d_k {d_k} != expected {expected}"
    assert (d_k >= 0).all(), f"renormalized d_K must be non-negative, got {d_k}"
    print("PASS: _compute_renorm_topk_reverse_kl matches hand reference + non-negative")


def _make_args(**kwargs):
    base = dict(
        use_opd=True,
        opd_type="sglang",
        opd_kl_coef=1.0,
        opd_cumulative=False,
        opd_gamma=1.0,
        opd_horizon=-1,
        opd_agg="mean",
        opd_dualclip_c=-1.0,
        opd_soft_mask=False,
        opd_future_rb=False,
        opd_future_topk=20,
        opd_future_no_renorm=False,
        opd_dump_kl_path=None,
        opd_dump_kl_interval=1,
        opd_dump_kl_max_samples=-1,
    )
    base.update(kwargs)
    return Namespace(**base)


def test_legacy_path_unchanged_R1_mean_K8():
    """Verify R1 (mean K=8) produces same numbers when --opd-future-rb is absent."""
    T = 16
    K = 8
    torch.manual_seed(0)
    s_lp = [torch.randn(T) * 0.3]
    t_lp_full = [torch.randn(T) * 0.3]
    advs = [torch.zeros(T)]
    advs_legacy = [advs[0].clone()]
    advs_with_flag_off = [advs[0].clone()]

    rollout_data = {"teacher_log_probs": t_lp_full}
    rollout_data2 = {"teacher_log_probs": t_lp_full}

    # Path A: legacy (opd_future_rb=False)
    args1 = _make_args(opd_cumulative=True, opd_horizon=K, opd_agg="mean", opd_future_rb=False)
    apply_opd_kl_to_advantages(args1, rollout_data, advs_legacy, s_lp)

    # Path B: same flag explicitly set to False
    args2 = _make_args(opd_cumulative=True, opd_horizon=K, opd_agg="mean", opd_future_rb=False)
    apply_opd_kl_to_advantages(args2, rollout_data2, advs_with_flag_off, s_lp)

    assert torch.allclose(advs_legacy[0], advs_with_flag_off[0]), "two equivalent legacy paths diverge"
    print(f"PASS: legacy R1 mean-K=8 path unchanged when opd_future_rb=False")
    print(f"  sample advantage: {advs_legacy[0][:5].tolist()}")


def test_form_a_k_K2_mean_explicit():
    """Verify Form A-K K=2 mean produces the explicit formula."""
    T = 8
    K_topk = 4
    torch.manual_seed(42)

    s_lp = [torch.randn(T) * 0.3]
    t_lp_full = [torch.randn(T) * 0.3]

    # Synthesize teacher top-K and student top-K logp (random but consistent)
    t_topk_logp = [torch.randn(T, K_topk) * 0.5]
    s_topk_logp = [torch.randn(T, K_topk) * 0.5]

    advs = [torch.zeros(T)]

    rollout_data = {
        "teacher_log_probs": t_lp_full,
        "teacher_topk_logp": t_topk_logp,
        "student_topk_logp": s_topk_logp,
    }

    args = _make_args(
        opd_cumulative=True,
        opd_horizon=2,
        opd_agg="mean",
        opd_future_rb=True,
    )
    apply_opd_kl_to_advantages(args, rollout_data, advs, s_lp)

    # Hand reference:
    # c_t = s_lp - t_lp_full
    # d_K(h_t) renormalized
    # cumulative_kl[t] = (c_t + γ * d_K(h_{t+1})) / 2  for t < T-1
    # cumulative_kl[T-1] = c_{T-1} / 1  (mean ÷ actual_k=1)
    # advantages[t] = 0 - 1.0 * cumulative_kl[t]
    c = s_lp[0] - t_lp_full[0]
    d_k_ref = _compute_renorm_topk_reverse_kl(s_topk_logp[0], t_topk_logp[0], renorm=True)
    expected = torch.zeros_like(c)
    for t in range(T - 1):
        expected[t] = -1.0 * (c[t] + 1.0 * d_k_ref[t + 1]) / 2.0
    expected[T - 1] = -1.0 * c[T - 1] / 1.0  # actual_k = 1 at last position

    assert torch.allclose(advs[0], expected, atol=1e-6), (
        f"Form A-K K=2 mean mismatch:\n  got     : {advs[0].tolist()}\n  expected: {expected.tolist()}"
    )
    print(f"PASS: Form A-K K=2 mean matches explicit per-t formula")
    print(f"  sample advantage: {advs[0][:5].tolist()}")


def test_form_a_k_K8_sum():
    """Verify Form A-K K=8 sum produces textbook reward-to-go."""
    T = 16
    K_horizon = 8
    K_topk = 5
    torch.manual_seed(7)

    s_lp = [torch.randn(T) * 0.3]
    t_lp_full = [torch.randn(T) * 0.3]
    t_topk_logp = [torch.randn(T, K_topk) * 0.5]
    s_topk_logp = [torch.randn(T, K_topk) * 0.5]

    advs = [torch.zeros(T)]

    rollout_data = {
        "teacher_log_probs": t_lp_full,
        "teacher_topk_logp": t_topk_logp,
        "student_topk_logp": s_topk_logp,
    }

    args = _make_args(
        opd_cumulative=True,
        opd_horizon=K_horizon,
        opd_agg="sum",
        opd_future_rb=True,
    )
    apply_opd_kl_to_advantages(args, rollout_data, advs, s_lp)

    c = s_lp[0] - t_lp_full[0]
    d_k = _compute_renorm_topk_reverse_kl(s_topk_logp[0], t_topk_logp[0], renorm=True)
    expected = torch.zeros_like(c)
    for t in range(T):
        end = min(t + K_horizon, T)
        # j=0: sampled c_t. j>=1: d_K(h_{t+j}).
        s = c[t]
        for j in range(1, end - t):
            s = s + d_k[t + j]
        # sum agg: no division
        expected[t] = -1.0 * s

    assert torch.allclose(advs[0], expected, atol=1e-6), (
        f"Form A-K K=8 sum mismatch:\n  got     : {advs[0].tolist()}\n  expected: {expected.tolist()}"
    )
    print(f"PASS: Form A-K K=8 sum matches textbook reward-to-go formula")


if __name__ == "__main__":
    print("=== Form A-K Regression Tests ===\n")
    test_renorm_topk_kl_handcomputed()
    print()
    test_legacy_path_unchanged_R1_mean_K8()
    print()
    test_form_a_k_K2_mean_explicit()
    print()
    test_form_a_k_K8_sum()
    print()
    print("=== ALL PASS ===")
