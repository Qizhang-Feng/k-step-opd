"""Standalone (no megatron) test for Form A-K math.

Reproduces the math of `_compute_renorm_topk_reverse_kl` and the K=2 mean / K=8 sum
formulas from `apply_opd_kl_to_advantages`'s Form A-K branch, using ONLY torch.

Cannot import slime here because that pulls in megatron — but the code we test
is the SAME math, copied verbatim from loss.py. The point is:
  - Sanity-check the formulas in isolation
  - Provide a reference for the in-cluster regression test (test_opd_future_rb_regression.py)
"""
import torch


def _compute_renorm_topk_reverse_kl(student_topk_logp, teacher_topk_logp, *, renorm=True):
    """Verbatim copy of slime/.../loss.py:_compute_renorm_topk_reverse_kl."""
    if renorm:
        s_norm_logp = student_topk_logp - torch.logsumexp(student_topk_logp, dim=-1, keepdim=True)
        t_norm_logp = teacher_topk_logp - torch.logsumexp(teacher_topk_logp, dim=-1, keepdim=True)
        d_k = (s_norm_logp.exp() * (s_norm_logp - t_norm_logp)).sum(dim=-1)
    else:
        d_k = (student_topk_logp.exp() * (student_topk_logp - teacher_topk_logp)).sum(dim=-1)
    return d_k


def apply_form_a_k_mean(c, d_k, K=2):
    """K=2 mean: cumulative_kl[t] = (c_t + d_K(h_{t+1})) / actual_k.

    actual_k = K for t < T - K + 1, then shrinks to 1 at the last position.
    For K=2 this means: cumulative_kl[t] = (c_t + d_K(h_{t+1})) / 2 for t < T-1,
    cumulative_kl[T-1] = c_{T-1} / 1.
    """
    T = c.size(0)
    cumkl = torch.zeros_like(c)
    for t in range(T):
        end = min(t + K, T)
        actual_k = end - t
        # j=0: sampled c_t; j>=1: d_K(h_{t+j})
        s = c[t]
        for j in range(1, actual_k):
            s = s + d_k[t + j]
        cumkl[t] = s / actual_k
    return cumkl


def apply_form_a_k_sum(c, d_k, K):
    """K=K sum: cumulative_kl[t] = c_t + sum_{j=1..K-1} d_K(h_{t+j})."""
    T = c.size(0)
    cumkl = torch.zeros_like(c)
    for t in range(T):
        end = min(t + K, T)
        s = c[t]
        for j in range(1, end - t):
            s = s + d_k[t + j]
        cumkl[t] = s
    return cumkl


def test_renorm_topk_kl():
    s_lp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]])
    t_lp = torch.tensor([[-1.5, -1.8, -2.5], [-1.0, -1.0, -3.0]])

    d_k = _compute_renorm_topk_reverse_kl(s_lp, t_lp, renorm=True)

    # Hand-compute position 0
    s0 = s_lp[0]
    t0 = t_lp[0]
    s0_norm = s0 - torch.logsumexp(s0, dim=-1)
    t0_norm = t0 - torch.logsumexp(t0, dim=-1)
    expected_0 = (s0_norm.exp() * (s0_norm - t0_norm)).sum()

    assert torch.allclose(d_k[0], expected_0), f"d_k[0]={d_k[0]:.6f} vs expected={expected_0:.6f}"
    assert (d_k >= 0).all(), f"renormalized KL must be ≥ 0 but got {d_k}"

    # Sum to 1 sanity
    assert torch.allclose(s0_norm.exp().sum(), torch.tensor(1.0))
    assert torch.allclose(t0_norm.exp().sum(), torch.tensor(1.0))

    print(f"PASS: _compute_renorm_topk_reverse_kl correct")
    print(f"  d_K = {d_k.tolist()}, all >= 0")


def test_K2_mean_boundary():
    """At the last position (t=T-1), actual_k=1 so cumkl[T-1] = c_{T-1}."""
    T = 5
    K_topk = 4
    torch.manual_seed(0)
    c = torch.randn(T) * 0.3
    s_topk = torch.randn(T, K_topk) * 0.5
    t_topk = torch.randn(T, K_topk) * 0.5
    d_k = _compute_renorm_topk_reverse_kl(s_topk, t_topk)

    cumkl = apply_form_a_k_mean(c, d_k, K=2)

    # Manual reference
    expected = torch.zeros_like(c)
    for t in range(T - 1):
        expected[t] = (c[t] + d_k[t + 1]) / 2.0
    expected[T - 1] = c[T - 1] / 1.0  # actual_k = 1

    assert torch.allclose(cumkl, expected, atol=1e-6), f"cumkl={cumkl} expected={expected}"
    print(f"PASS: K=2 mean produces correct boundary-handled formula")
    print(f"  cumkl   = {cumkl.tolist()}")
    print(f"  expected= {expected.tolist()}")


def test_K8_sum():
    """K=8 sum: textbook reward-to-go without normalization."""
    T = 16
    K_horizon = 8
    K_topk = 5
    torch.manual_seed(7)
    c = torch.randn(T) * 0.3
    s_topk = torch.randn(T, K_topk) * 0.5
    t_topk = torch.randn(T, K_topk) * 0.5
    d_k = _compute_renorm_topk_reverse_kl(s_topk, t_topk)

    cumkl = apply_form_a_k_sum(c, d_k, K=K_horizon)

    expected = torch.zeros_like(c)
    for t in range(T):
        end = min(t + K_horizon, T)
        s = c[t]
        for j in range(1, end - t):
            s = s + d_k[t + j]
        expected[t] = s

    assert torch.allclose(cumkl, expected, atol=1e-6)
    # spot check: t=0 should sum c_0 + d_K(1) + d_K(2) + ... + d_K(7)
    expected_t0 = c[0] + d_k[1:8].sum()
    assert torch.allclose(cumkl[0], expected_t0, atol=1e-6)
    print(f"PASS: K=8 sum textbook reward-to-go formula correct")
    print(f"  cumkl[0] = c[0] + sum(d_K[1:8]) = {cumkl[0]:.4f}")


def test_form_a_k_vs_legacy_when_d_k_equals_c():
    """Sanity: if d_K were equal to c at every position (hypothetical), Form A-K
    should reduce to the legacy mean-K cumulative sum exactly. This is a *design*
    sanity test — when the future RB term equals the future sampled term, we're
    not changing anything. Useful to verify cumulative-sum logic is correct.
    """
    T = 16
    K = 8
    torch.manual_seed(99)
    c = torch.randn(T) * 0.3
    # Set d_k = c (hypothetical alignment; not what we use in practice).
    d_k = c.clone()

    cumkl_form_a = apply_form_a_k_mean(c, d_k, K=K)

    # Legacy mean-K (what slime currently does for R1 with sampled future):
    expected_legacy = torch.zeros_like(c)
    for t in range(T):
        end = min(t + K, T)
        actual_k = end - t
        expected_legacy[t] = c[t:end].sum() / actual_k

    assert torch.allclose(cumkl_form_a, expected_legacy, atol=1e-6), (
        f"Form A-K with d_k = c should match legacy: \n"
        f"  form_a  = {cumkl_form_a}\n  legacy  = {expected_legacy}"
    )
    print(f"PASS: Form A-K with d_k = c reduces to legacy mean-K (correctness of cumulative-sum logic)")


if __name__ == "__main__":
    print("=== Form A-K Standalone Math Tests ===\n")
    test_renorm_topk_kl()
    print()
    test_K2_mean_boundary()
    print()
    test_K8_sum()
    print()
    test_form_a_k_vs_legacy_when_d_k_equals_c()
    print()
    print("=== ALL PASS ===")
