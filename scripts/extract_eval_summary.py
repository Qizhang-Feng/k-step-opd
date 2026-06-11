"""Extract avg_pass_at_1 from a directory of n=16 eval JSON files.

Reads each JSON in `--dir` (skips `.partial`), parses the per-problem
`pass_at_k_samples` and computes avg_pass_at_1 = mean over all problems of
the fraction of correct samples per problem.

Usage:
    python3 extract_eval_summary.py --dir /opt/dlami/nvme/qzf/k-step-opd/eval_results_n16
"""
import argparse
import json
import os
import sys
from glob import glob


def parse_one(path):
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict) or "avg_pass_at_1" not in d:
        return None
    return {
        "n_problems": d.get("n_problems"),
        "n_samples": d.get("n_samples"),
        "avg_pass_at_1": d.get("avg_pass_at_1"),
        "avg_first_boxed_pass_at_1": d.get("avg_first_boxed_pass_at_1"),
        "pass_any": d.get("pass_at_any") or d.get("first_boxed_pass_at_any"),
        "avg_len": d.get("avg_response_length"),
        "truncation_rate": d.get("truncation_rate"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    files = [f for f in sorted(glob(os.path.join(args.dir, "*.json"))) if not f.endswith(".partial")]
    rows = []
    for f in files:
        name = os.path.basename(f)
        if not (name.startswith("aime2024_") or name.startswith("aime2025_")):
            continue
        bench = "AIME-24" if name.startswith("aime2024_") else "AIME-25"
        run_id = name[len("aime2024_") :].replace(".json", "")
        s = parse_one(f)
        if s:
            rows.append((run_id, bench, s))

    # Group by run_id
    runs = {}
    for run_id, bench, s in rows:
        runs.setdefault(run_id, {})[bench] = s

    # Print sorted
    print(f"{'run':50s} {'AIME-24':>9s} {'p_any-24':>9s} {'AIME-25':>9s} {'p_any-25':>9s} {'len-24':>8s} {'#samples':>9s}")
    for run_id in sorted(runs):
        d = runs[run_id]
        a24 = d.get("AIME-24")
        a25 = d.get("AIME-25")

        def fmt(v):
            if v is None:
                return "      —  "
            return f"{v * 100:>7.2f}%"

        avg_len = (a24 or {}).get("avg_len") or (a25 or {}).get("avg_len")
        n_samples = (a24 or {}).get("n_samples") or (a25 or {}).get("n_samples")
        len_str = f"{avg_len:>8.0f}" if avg_len else "       —"
        n_str = f"{n_samples:>9d}" if n_samples else "        —"
        print(
            f"{run_id:50s} "
            f"{fmt((a24 or {}).get('avg_pass_at_1')):>9s} "
            f"{fmt((a24 or {}).get('pass_any')):>9s} "
            f"{fmt((a25 or {}).get('avg_pass_at_1')):>9s} "
            f"{fmt((a25 or {}).get('pass_any')):>9s} "
            f"{len_str:>8s} "
            f"{n_str:>9s}"
        )


if __name__ == "__main__":
    main()
