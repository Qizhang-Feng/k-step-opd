"""
Standalone math eval script using SGLang server + slime's deepscaler reward.

Usage:
    python eval_math.py \
        --server-url http://127.0.0.1:30000 \
        --data-path /workspace/data/math-500/math-500.jsonl \
        --output-path results.json \
        --n-samples 8 \
        --max-tokens 16384 \
        --temperature 0.6

Data format (jsonl): {"prompt": "...", "label": "42", "reward_model": "math"}
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Add slime to path for reward function
sys.path.insert(0, "/root/slime")
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from slime.rollout.rm_hub.math_utils import extract_answer


def load_data(path):
    """Load eval data from jsonl."""
    data = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            data.append(d)
    return data


def generate_one(server_url, prompt, n_samples, max_tokens, temperature):
    """Generate n_samples responses for a single prompt via SGLang.

    Splits into n_samples independent n=1 requests run in parallel.
    This way SGLang's DP load balancer distributes samples across replicas,
    avoiding single-replica KV overflow when n*max_tokens >> max_total_tokens.
    """
    def _one_call():
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "n": 1,
            },
        }
        resp = requests.post(
            f"{server_url}/generate",
            json=payload,
            timeout=1800,
        )
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return result[0].get("text", "")
        return result.get("text", "")

    if n_samples == 1:
        return [_one_call()]

    from concurrent.futures import ThreadPoolExecutor as _TPE
    texts = [""] * n_samples
    with _TPE(max_workers=n_samples) as ex:
        for i, t in enumerate(ex.map(lambda _: _one_call(), range(n_samples))):
            texts[i] = t
    return texts


def apply_chat_template(prompt, model_type="qwen"):
    """Wrap prompt in chat template."""
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n"
    )


def grade_response(response, label):
    """Grade a single response. Returns (strict_reward, loose_reward, first_boxed_reward)."""
    # Strict: deepscaler (requires </think>)
    strict = get_deepscaler_rule_based_reward(response, label)

    # Loose: extract last \boxed{} from full response (current behavior)
    loose = 0
    answer = extract_answer(response)
    if answer is not None:
        from slime.rollout.rm_hub.deepscaler import grade_answer_mathd, grade_answer_sympy
        truth = str(label)
        if "\\boxed" in truth:
            processed = extract_answer(truth)
            if processed is not None:
                truth = processed
        if grade_answer_mathd(answer, truth) or grade_answer_sympy(answer, truth):
            loose = 1

    # First boxed: extract first \boxed{} only
    first_boxed = 0
    boxed_pos = response.find("\\boxed{")
    if boxed_pos >= 0:
        # Extract just the first boxed segment
        first_segment = response[boxed_pos:boxed_pos+200]
        first_answer = extract_answer(first_segment)
        if first_answer is not None:
            from slime.rollout.rm_hub.deepscaler import grade_answer_mathd, grade_answer_sympy
            truth = str(label)
            if "\\boxed" in truth:
                processed = extract_answer(truth)
                if processed is not None:
                    truth = processed
            if grade_answer_mathd(first_answer, truth) or grade_answer_sympy(first_answer, truth):
                first_boxed = 1

    # Return best of strict/loose for backward compat, but track all three
    return max(strict, loose), first_boxed


def majority_vote(responses, label):
    """Compute majority vote accuracy from multiple responses."""
    answers = []
    for resp in responses:
        # Extract answer from response
        if "</think>" in resp:
            solution = resp.split("</think>")[-1]
        else:
            solution = resp
        answer = extract_answer(solution)
        if answer is not None:
            answers.append(answer)

    if not answers:
        return 0

    # Most common answer
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]

    # Check if most common answer matches label
    # Use deepscaler grading on a synthetic response
    synthetic = f"Answer: \\boxed{{{most_common}}}"
    return get_deepscaler_rule_based_reward(synthetic, label)


def process_one_problem(args_tuple):
    """Process a single problem: generate + grade. For use with ThreadPoolExecutor."""
    i, item, server_url, n_samples, max_tokens, temperature = args_tuple
    prompt = item["prompt"]
    label = item["label"]
    formatted_prompt = apply_chat_template(prompt)

    try:
        responses = generate_one(server_url, formatted_prompt, n_samples, max_tokens, temperature)
    except Exception as e:
        return {
            "idx": i,
            "prompt": prompt[:100],
            "label": label,
            "error": str(e),
            "rewards": [],
            "pass1": 0,
            "pass_any": 0,
            "majority": 0,
            "truncated": 0,
            "n_truncated": 0,
        }

    rewards = []
    first_boxed_rewards = []
    n_truncated = 0
    for resp in responses:
        r, fb = grade_response(resp, label)
        rewards.append(r)
        first_boxed_rewards.append(fb)
        # Detect truncation: response hit max_tokens without proper ending
        # Heuristic: no \boxed{} and no </think> near end means likely truncated
        if "\\boxed" not in resp and len(resp) > max_tokens * 3:
            n_truncated += 1
    # avg_pass1 = mean over all n samples (Lightning-OPD style "average pass@1")
    # pass1_first = legacy: 1 if the first sample is correct (kept for back-compat)
    n = max(len(rewards), 1)
    avg_pass1 = sum(rewards) / n
    avg_first_boxed = sum(first_boxed_rewards) / n
    pass1_first = 1 if rewards and rewards[0] > 0 else 0
    pass_any = 1 if any(r > 0 for r in rewards) else 0
    first_boxed_pass1_first = 1 if first_boxed_rewards and first_boxed_rewards[0] > 0 else 0
    first_boxed_any = 1 if any(r > 0 for r in first_boxed_rewards) else 0
    maj = majority_vote(responses, label)

    return {
        "idx": i,
        "prompt": prompt[:200],
        "label": label,
        "rewards": rewards,
        "first_boxed_rewards": first_boxed_rewards,
        "avg_pass1": avg_pass1,                       # mean over n samples
        "avg_first_boxed": avg_first_boxed,           # mean over n samples
        "pass1": pass1_first,                          # back-compat: first sample only
        "pass_any": pass_any,
        "first_boxed_pass1": first_boxed_pass1_first,  # back-compat: first sample only
        "first_boxed_any": first_boxed_any,
        "majority": maj,
        "avg_len": sum(len(r) for r in responses) / len(responses),
        "truncated": 1 if n_truncated > 0 else 0,
        "n_truncated": n_truncated,
        "response_full": responses[0],  # save full first response
        "responses_preview": [r[:500] for r in responses[1:]],  # preview of others
    }


def eval_dataset(server_url, data, n_samples, max_tokens, temperature, dataset_name, max_workers=16, output_path=None):
    """Evaluate a dataset with parallel requests."""
    results = [None] * len(data)
    correct_pass1 = 0
    correct_any = 0
    correct_majority = 0
    correct_first_boxed_pass1 = 0
    correct_first_boxed_any = 0
    sum_avg_pass1 = 0.0           # accumulated mean-over-n pass@1 (per problem)
    sum_avg_first_boxed = 0.0     # accumulated mean-over-n first_boxed (per problem)
    total_length = 0
    total_responses = 0
    completed = 0

    print(f"Evaluating {len(data)} problems, {n_samples} samples each, {max_workers} parallel workers...")

    task_args = [
        (i, item, server_url, n_samples, max_tokens, temperature)
        for i, item in enumerate(data)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_problem, args): args[0] for args in task_args}
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result

            correct_pass1 += result["pass1"]
            correct_any += result["pass_any"]
            correct_majority += result["majority"]
            correct_first_boxed_pass1 += result.get("first_boxed_pass1", 0)
            correct_first_boxed_any += result.get("first_boxed_any", 0)
            sum_avg_pass1 += result.get("avg_pass1", 0.0)
            sum_avg_first_boxed += result.get("avg_first_boxed", 0.0)
            if "avg_len" in result:
                total_length += result["avg_len"] * n_samples
                total_responses += n_samples

            completed += 1
            if completed % 5 == 0 or completed == len(data):
                avg_p1 = sum_avg_pass1 / completed
                pa = correct_any / completed
                fb_avg = sum_avg_first_boxed / completed
                print(
                    f"  [{completed}/{len(data)}] "
                    f"avg_pass@1={avg_p1:.3f} pass@any={pa:.3f} first_boxed_avg={fb_avg:.3f}",
                    flush=True,
                )
                # Save intermediate results
                if output_path:
                    partial = {
                        "dataset_name": dataset_name,
                        "completed": completed,
                        "n_problems": len(data),
                        "avg_pass_at_1": sum_avg_pass1 / completed,
                        "avg_first_boxed_pass_at_1": sum_avg_first_boxed / completed,
                        "pass_at_1": correct_pass1 / completed,    # back-compat (1st sample only)
                        "pass_at_any": correct_any / completed,
                        "majority_vote": correct_majority / completed,
                        "avg_response_length": total_length / max(total_responses, 1),
                        "details": [r for r in results if r is not None],
                    }
                    with open(output_path + ".partial", "w") as f:
                        json.dump(partial, f, indent=2, ensure_ascii=False)

    n = len(data)
    avg_len = total_length / max(total_responses, 1)
    total_truncated = sum(1 for r in results if r and r.get("truncated", 0))

    summary = {
        "dataset_name": dataset_name,
        "n_problems": n,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # NEW: average pass@1 over n samples (Lightning-OPD reporting style)
        "avg_pass_at_1": sum_avg_pass1 / n if n > 0 else 0,
        "avg_first_boxed_pass_at_1": sum_avg_first_boxed / n if n > 0 else 0,
        # Back-compat (first sample only — equivalent to n=1 eval)
        "pass_at_1": correct_pass1 / n if n > 0 else 0,
        "pass_at_any": correct_any / n if n > 0 else 0,
        "majority_vote": correct_majority / n if n > 0 else 0,
        "first_boxed_pass_at_1": correct_first_boxed_pass1 / n if n > 0 else 0,
        "first_boxed_pass_at_any": correct_first_boxed_any / n if n > 0 else 0,
        "avg_response_length": avg_len,
        "truncation_rate": total_truncated / n if n > 0 else 0,
        "n_truncated": total_truncated,
        "details": results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Math eval using SGLang + deepscaler reward")
    parser.add_argument("--server-url", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dataset-name", type=str, default="math500")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=16, help="Parallel request workers")
    args = parser.parse_args()

    # Load data
    data = load_data(args.data_path)
    print(f"Loaded {len(data)} problems from {args.data_path}")

    # Run eval
    t0 = time.time()
    summary = eval_dataset(
        args.server_url,
        data,
        args.n_samples,
        args.max_tokens,
        args.temperature,
        args.dataset_name,
        max_workers=args.max_workers,
        output_path=args.output_path,
    )
    elapsed = time.time() - t0

    # Add metadata
    summary["model_name"] = args.model_name or os.path.basename(args.output_path).replace(".json", "")
    summary["elapsed_seconds"] = elapsed

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset_name} ({summary['n_problems']} problems, n={args.n_samples})")
    print(f"avg pass@1:    {summary['avg_pass_at_1']:.4f} ({summary['avg_pass_at_1']*100:.1f}%)  ← noise-reduced (mean over n)")
    print(f"avg fb@1:      {summary['avg_first_boxed_pass_at_1']:.4f} ({summary['avg_first_boxed_pass_at_1']*100:.1f}%)")
    print(f"pass@1 [s0]:   {summary['pass_at_1']:.4f} ({summary['pass_at_1']*100:.1f}%)  ← back-compat (1st sample)")
    print(f"pass@any:      {summary['pass_at_any']:.4f} ({summary['pass_at_any']*100:.1f}%)")
    print(f"maj@{args.n_samples}:        {summary['majority_vote']:.4f} ({summary['majority_vote']*100:.1f}%)")
    print(f"Avg length:    {summary['avg_response_length']:.0f} chars")
    print(f"Time:          {elapsed:.0f}s")
    print(f"Saved to:      {args.output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
