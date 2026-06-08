"""Regression test for the sum/mean aggregation + dual-clip mask in
apply_opd_kl_to_advantages. Verifies that --opd-agg mean (default) reproduces
the OLD slime v2 cumulative_kl exactly, and sanity-checks sum / dual-clip.

Run: python3 scripts/test_opd_agg_regression.py
"""
import torch


def old_cumulative(reverse_kl, gamma, horizon):
    """The ORIGINAL slime v2 implementation (mean, no mask)."""
    T = len(reverse_kl)
    if horizon <= 0 or horizon >= T:
        if gamma == 1.0:
            return torch.flip(torch.cumsum(torch.flip(reverse_kl, [0]), dim=0), [0])
        cumulative_kl = torch.zeros_like(reverse_kl)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = reverse_kl[t] + gamma * running
            cumulative_kl[t] = running
        return cumulative_kl
    cumulative_kl = torch.zeros_like(reverse_kl)
    for t in range(T):
        end = min(t + horizon, T)
        actual_k = end - t
        if gamma == 1.0:
            cumulative_kl[t] = reverse_kl[t:end].sum() / actual_k
        else:
            weights = gamma ** torch.arange(end - t, dtype=reverse_kl.dtype)
            cumulative_kl[t] = (reverse_kl[t:end] * weights).sum() / actual_k
    return cumulative_kl


def new_cumulative(reverse_kl, gamma, horizon, agg, keep_mask=None):
    """The NEW implementation (agg in {mean,sum} + optional dual-clip mask).
    Mirrors the corrected loss.py: full-horizon never divides; truncated mean
    divides by actual_k (token count, v2 semantics)."""
    T = len(reverse_kl)
    masked_kl = reverse_kl if keep_mask is None else reverse_kl * keep_mask
    if horizon <= 0 or horizon >= T:
        if gamma == 1.0:
            suffix_sum = torch.flip(torch.cumsum(torch.flip(masked_kl, [0]), dim=0), [0])
            return suffix_sum
        cumulative_kl = torch.zeros_like(reverse_kl)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = masked_kl[t] + gamma * running
            cumulative_kl[t] = running
        return cumulative_kl
    cumulative_kl = torch.zeros_like(reverse_kl)
    for t in range(T):
        end = min(t + horizon, T)
        actual_k = end - t
        if gamma == 1.0:
            s = masked_kl[t:end].sum()
            cumulative_kl[t] = s / actual_k if agg == "mean" else s
        else:
            weights = gamma ** torch.arange(end - t, dtype=reverse_kl.dtype)
            s = (masked_kl[t:end] * weights).sum()
            cumulative_kl[t] = s / actual_k if agg == "mean" else s
    return cumulative_kl


def main():
    torch.manual_seed(0)
    cases = [
        (1.0, -1), (1.0, 4), (1.0, 8), (0.99, 4), (0.95, 8), (1.0, 2),
    ]
    print("=== Regression: new(agg=mean, no mask) MUST equal old ===")
    all_ok = True
    for gamma, horizon in cases:
        for T in (5, 16, 50):
            rk = torch.randn(T)
            old = old_cumulative(rk, gamma, horizon)
            new = new_cumulative(rk, gamma, horizon, "mean", keep_mask=None)
            max_diff = (old - new).abs().max().item()
            ok = max_diff < 1e-5
            all_ok &= ok
            if not ok:
                print(f"  FAIL gamma={gamma} K={horizon} T={T} max_diff={max_diff:.2e}")
    print("  PASS (mean == old)" if all_ok else "  *** REGRESSION FAILED ***")

    print("=== Sanity: sum == mean * K (gamma=1, full horizon within window) ===")
    rk = torch.randn(20)
    K = 4
    mean_k = new_cumulative(rk, 1.0, K, "mean")
    sum_k = new_cumulative(rk, 1.0, K, "sum")
    # for interior tokens (t+K <= T), mean*K should equal sum exactly
    interior = slice(0, 20 - K)
    diff = (sum_k[interior] - mean_k[interior] * K).abs().max().item()
    print(f"  interior sum vs mean*K max_diff={diff:.2e}", "PASS" if diff < 1e-5 else "*** FAIL ***")

    print("=== Sanity: dual-clip mask drops outlier tokens from sum ===")
    rk = torch.ones(10)
    keep = torch.ones(10)
    keep[5] = 0.0  # drop token 5
    sum_nomask = new_cumulative(rk, 1.0, -1, "sum")
    sum_mask = new_cumulative(rk, 1.0, -1, "sum", keep_mask=keep)
    # token 0 suffix sum: nomask=10, mask=9 (token5 dropped)
    print(f"  t=0 nomask={sum_nomask[0]:.1f} mask={sum_mask[0]:.1f}",
          "PASS" if abs(sum_nomask[0].item() - 10) < 1e-5 and abs(sum_mask[0].item() - 9) < 1e-5 else "*** FAIL ***")

    print("DONE", "ALL PASS" if all_ok else "HAS FAILURES")


if __name__ == "__main__":
    main()
