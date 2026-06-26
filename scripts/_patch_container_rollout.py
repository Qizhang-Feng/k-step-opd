"""Surgical patch: add teacher_topk_ids/teacher_topk_logp to slime/ray/rollout.py.

The workspace slime is newer than the slime baked into the container image
(`slime.backends.sglang_utils.sglang_config` doesn't exist in the container).
We can't replace rollout.py wholesale; we patch in place.

Run inside the slime container:
    python3 /workspace/k-step-opd/scripts/_patch_container_rollout.py
"""
import sys

path = "/root/slime/slime/ray/rollout.py"
src = open(path).read()

# Idempotency guard — if already patched, skip.
if "teacher_topk_ids" in src:
    print("already patched (teacher_topk_ids already present); no-op")
    sys.exit(0)

# 1. Add to _build_train_data after teacher_log_probs append
needle1 = '        if samples[0].teacher_log_probs is not None:\n            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]\n\n        return train_data'
addition1 = '''        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        # Form A-K: teacher top-K logprobs + token ids (per response position).
        if samples[0].teacher_topk_ids is not None:
            train_data["teacher_topk_ids"] = [sample.teacher_topk_ids for sample in samples]
            train_data["teacher_topk_logp"] = [sample.teacher_topk_logp for sample in samples]

        return train_data'''

if needle1 in src:
    src = src.replace(needle1, addition1, 1)
    print("OK: applied _build_train_data block")
else:
    print("FAIL: _build_train_data needle missing — please inspect the file manually")
    sys.exit(1)

# 2. Add to _split_train_data_by_dp keys list
needle2 = '"teacher_log_probs",\n            ]:'
addition2 = '"teacher_log_probs",\n                "teacher_topk_ids",\n                "teacher_topk_logp",\n            ]:'
if needle2 in src:
    src = src.replace(needle2, addition2, 1)
    print("OK: applied keys list block")
else:
    print("FAIL: keys list needle missing — please inspect the file manually")
    sys.exit(1)

open(path, "w").write(src)
print("DONE")
