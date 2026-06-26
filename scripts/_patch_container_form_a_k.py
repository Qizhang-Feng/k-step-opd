"""Surgical patches to add Form A-K (hybrid OPD with RB future) to container slime.

Adapted to the *container's* slime version (not the workspace fork).
Notable container-specific differences vs workspace:
- get_log_probs_and_entropy uses per-sample loop with get_responses generator,
  not the workspace's "build_shifted_tokens + one fused CE call" pattern.
  This actually makes inserting student top-K gather simpler (per-sample inner loop).
- apply_opd_kl_to_advantages docstring + IS/soft-weight diagnostics differ from workspace.
- ray/rollout.py imports `slime.backends.sglang_utils.sglang_config` (not in workspace).

Idempotent: re-running skips already-applied patches.

Run inside the slime container:
    docker exec k-step-opd python3 /workspace/k-step-opd/_phase3_staging/_patch_container_form_a_k.py
"""
import sys
from pathlib import Path

PATCHES = []


def patch(path: str, marker: str, locator: str, replacement: str, name: str):
    PATCHES.append({"path": path, "marker": marker, "locator": locator, "replacement": replacement, "name": name})


# ============================================================================
# types.py — add Sample fields for teacher_topk_ids/logp
# ============================================================================
patch(
    path="/root/slime/slime/utils/types.py",
    marker="teacher_topk_ids",
    locator='    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD',
    replacement="""    teacher_log_probs: list[float] | None = None  # Log probabilities from teacher model for OPD
    # Form A-K (hybrid OPD): teacher top-K logprobs + token ids per response position.
    teacher_topk_ids: list[list[int]] | None = None
    teacher_topk_logp: list[list[float]] | None = None""",
    name="types.py: add teacher_topk_ids/logp Sample fields",
)


# ============================================================================
# on_policy_distillation.py — full file replacement (file is ours, small)
# ============================================================================
ON_POLICY_NEW = '''import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    payload = {
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    # Form A-K: request teacher top-K logprobs at every position so we can compute
    # truncated reverse-KL d_K(h_t) = KL(pi_S_renorm || pi_T_renorm) over the
    # teacher\'s top-K support set during training (paper arXiv 2603.25562 Eq. 6-8).
    opd_future_topk = getattr(args, "opd_future_topk", 0) or 0
    if getattr(args, "opd_future_rb", False) and opd_future_topk > 0:
        payload["top_logprobs_num"] = int(opd_future_topk)

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples, **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Form A-K: parse input_top_logprobs if requested.
    # Wire format: meta_info["input_top_logprobs"] : list[T_input] of (list[K] | None)
    # each top-K entry : [logprob, token_id, token_str_or_None]
    if getattr(args, "opd_future_rb", False):
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
            input_top = reward["meta_info"].get("input_top_logprobs")
            if input_top is None:
                sample.teacher_topk_ids = None
                sample.teacher_topk_logp = None
                continue

            response_top = input_top[1:][-response_length:]
            topk_ids = []
            topk_logp = []
            for pos in response_top:
                if pos is None:
                    topk_ids.append([0])
                    topk_logp.append([0.0])
                else:
                    topk_ids.append([entry[1] for entry in pos])
                    topk_logp.append([entry[0] for entry in pos])
            sample.teacher_topk_ids = topk_ids
            sample.teacher_topk_logp = topk_logp

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
'''
patch(
    path="/root/slime/slime/rollout/on_policy_distillation.py",
    marker="opd_future_topk",  # only present in patched version
    locator="<UNUSED-WHOLE-FILE-REPLACEMENT>",
    replacement=ON_POLICY_NEW,
    name="on_policy_distillation.py: full file replacement",
)


# ============================================================================
# arguments.py — add 3 Form A-K args
# ============================================================================
ARGS_LOCATOR = '''            parser.add_argument(
                "--opd-dump-kl-max-samples",
                type=int,
                default=-1,
                help=(
                    "Max samples per dump file (across all ranks combined). "
                    "-1 means dump all samples in the batch. Useful to limit disk usage."
                ),
            )
            return parser'''
ARGS_REPLACEMENT = '''            parser.add_argument(
                "--opd-dump-kl-max-samples",
                type=int,
                default=-1,
                help=(
                    "Max samples per dump file (across all ranks combined). "
                    "-1 means dump all samples in the batch. Useful to limit disk usage."
                ),
            )
            # ----- Form A-K: Hybrid OPD with Rao-Blackwellized future term -----
            # Math: A_t = -[ c_t + sum_{j=1..K-1} gamma^j * d_K(h_{t+j}) ] / (1 or K)
            #   c_t   = log pi_S(y_t|h_t) - log pi_T(y_t|h_t)   (sampled, action-specific)
            #   d_K(h) = KL(pi_S_renorm(.|h) || pi_T_renorm(.|h)) over teacher top-K support
            # Reference: arXiv 2603.25562 (Revisiting OPD) §3 — teacher top-K local support matching.
            #
            # This is a BIASED surrogate of the original sequence-level reverse-KL gradient
            # because of: (i) top-K truncation, (ii) finite K-step horizon, (iii) gamma<1.
            # Under (full vocab d, gamma=1, K=infty, on-policy) it would be unbiased.
            parser.add_argument(
                "--opd-future-rb",
                action="store_true",
                default=False,
                help=(
                    "Form A-K: replace future j>=1 sampled c_{t+j} in cumulative OPD "
                    "advantage with the Rao-Blackwellized teacher top-K renormalized "
                    "reverse-KL d_K(h_{t+j}). Requires teacher_topk_ids/logp populated "
                    "by the rollout. j=0 stays sampled (PG baseline trap forbids RB at t=0)."
                ),
            )
            parser.add_argument(
                "--opd-future-topk",
                type=int,
                default=20,
                help="Form A-K: top-K size for teacher local support set. Default 20 follows arXiv 2603.25562 §4.",
            )
            parser.add_argument(
                "--opd-future-no-renorm",
                action="store_true",
                default=False,
                help=(
                    "Form A-K: DISABLE renormalization within top-K support set when "
                    "computing d_K. Default behavior (flag absent) follows paper Eq. 7."
                ),
            )
            return parser'''
patch(
    path="/root/slime/slime/utils/arguments.py",
    marker='"--opd-future-rb"',
    locator=ARGS_LOCATOR,
    replacement=ARGS_REPLACEMENT,
    name="arguments.py: add 3 Form A-K args",
)


# ============================================================================
# loss.py — two changes
# ============================================================================
# Change 1: Add _compute_renorm_topk_reverse_kl helper near the top (before
# get_responses). Inserted right after the from .cp_utils import block.

LOSS_HELPER_LOCATOR = '''from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def get_responses('''

LOSS_HELPER_REPLACEMENT = '''from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)


def _compute_renorm_topk_reverse_kl(
    student_topk_logp: torch.Tensor,
    teacher_topk_logp: torch.Tensor,
    *,
    renorm: bool = True,
) -> torch.Tensor:
    """Compute Form A-K renormalized truncated reverse-KL d_K(h) per position.

    d_K(h_t) = KL( pi_S_renorm(.|h_t, S_t) || pi_T_renorm(.|h_t, S_t) )
    where S_t = teacher top-K support set at position t and pi_S_renorm /
    pi_T_renorm are the student / teacher distributions renormalized within S_t
    (paper Eq. 7, arXiv 2603.25562).

    Args:
        student_topk_logp: [T, K] full-vocab student log-probs at the K teacher-top-K
            token ids per position.
        teacher_topk_logp: [T, K] full-vocab teacher log-probs at the same K ids.
        renorm: If True (paper default), renormalize both distributions within
            the K-token support before computing KL.

    Returns:
        [T] tensor of d_K values per position. Non-negative when renorm=True.
    """
    if renorm:
        s_norm_logp = student_topk_logp - torch.logsumexp(student_topk_logp, dim=-1, keepdim=True)
        t_norm_logp = teacher_topk_logp - torch.logsumexp(teacher_topk_logp, dim=-1, keepdim=True)
        d_k = (s_norm_logp.exp() * (s_norm_logp - t_norm_logp)).sum(dim=-1)
    else:
        d_k = (student_topk_logp.exp() * (student_topk_logp - teacher_topk_logp)).sum(dim=-1)
    return d_k


def get_responses('''

patch(
    path="/root/slime/slime/backends/megatron_utils/loss.py",
    marker="_compute_renorm_topk_reverse_kl",
    locator=LOSS_HELPER_LOCATOR,
    replacement=LOSS_HELPER_REPLACEMENT,
    name="loss.py: add _compute_renorm_topk_reverse_kl helper",
)


# Change 2: Extend get_log_probs_and_entropy signature + per-sample loop to
# compute student top-K logp on teacher-selected support tokens.
LOSS_GLPE_LOCATOR = '''def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:'''
LOSS_GLPE_REPLACEMENT = '''def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    teacher_topk_ids: list[torch.Tensor] | None = None,
) -> dict[str, list[torch.Tensor]]:'''

patch(
    path="/root/slime/slime/backends/megatron_utils/loss.py",
    marker="teacher_topk_ids: list[torch.Tensor] | None",
    locator=LOSS_GLPE_LOCATOR,
    replacement=LOSS_GLPE_REPLACEMENT,
    name="loss.py: get_log_probs_and_entropy signature add teacher_topk_ids",
)

# Change 3: Per-sample inner loop add student top-K gather.
LOSS_GLPE_LOOP_LOCATOR = '''    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list'''

LOSS_GLPE_LOOP_REPLACEMENT = '''    log_probs_list = []
    entropy_list = []
    student_topk_list = []
    for i, (logits_chunk, tokens_chunk) in enumerate(get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            mpu.get_tensor_model_parallel_group(),
            with_entropy=with_entropy,
            chunk_size=args.log_probs_chunk_size,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

        # Form A-K: also compute student log-prob on each of the K teacher-top-K
        # token ids at each response position. K=20 small kernel calls; this is
        # forward_only / no_grad so no backward overhead.
        if teacher_topk_ids is not None and i < len(teacher_topk_ids):
            tk_ids = teacher_topk_ids[i].to(device=logits_chunk.device, dtype=torch.long)
            K = tk_ids.size(-1)
            topk_logps = []
            for k in range(K):
                lp_k, _ = calculate_log_probs_and_entropy(
                    logits_chunk,
                    tk_ids[:, k].contiguous(),
                    mpu.get_tensor_model_parallel_group(),
                    with_entropy=False,
                    chunk_size=args.log_probs_chunk_size,
                )
                topk_logps.append(lp_k.squeeze(-1))
            student_topk_list.append(torch.stack(topk_logps, dim=-1))  # [R, K]

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    if teacher_topk_ids is not None and len(student_topk_list) > 0:
        if args.allgather_cp:
            raise NotImplementedError(
                "Form A-K (--opd-future-rb) does not yet support allgather_cp; "
                "the [R, K] redistribute path is not implemented."
            )
        res["student_topk_logp"] = student_topk_list'''

patch(
    path="/root/slime/slime/backends/megatron_utils/loss.py",
    marker="student_topk_list",
    locator=LOSS_GLPE_LOOP_LOCATOR,
    replacement=LOSS_GLPE_LOOP_REPLACEMENT,
    name="loss.py: get_log_probs_and_entropy add student top-K gather loop",
)

# Change 4: apply_opd_kl_to_advantages cumulative branch — add Form A-K future
# RB substitution. Locator anchors on container-version's truncated-horizon block.
# We REPLACE the 'masked_kl = reverse_kl' branch + horizon block with one that
# computes d_K and uses it for j>=1 contributions when --opd-future-rb is set.
LOSS_APPLY_LOCATOR = '''            else:
                masked_kl = reverse_kl

            if horizon <= 0 or horizon >= T:
                # full sequence (v2 semantics: never divide; agg only matters for truncated)
                if gamma == 1.0:
                    cumulative_kl = torch.flip(torch.cumsum(torch.flip(masked_kl, [0]), dim=0), [0])
                else:
                    cumulative_kl = torch.zeros_like(reverse_kl)
                    running = torch.zeros((), dtype=reverse_kl.dtype, device=reverse_kl.device)
                    for t in range(T - 1, -1, -1):
                        running = masked_kl[t] + gamma * running
                        cumulative_kl[t] = running
            else:
                # truncated horizon: each token only looks K steps ahead
                cumulative_kl = torch.zeros_like(reverse_kl)
                for t in range(T):
                    end = min(t + horizon, T)
                    actual_k = end - t
                    if gamma == 1.0:
                        s_ = masked_kl[t:end].sum()
                    else:
                        weights = gamma ** torch.arange(actual_k, device=reverse_kl.device, dtype=reverse_kl.dtype)
                        s_ = (masked_kl[t:end] * weights).sum()
                    cumulative_kl[t] = (s_ / actual_k) if agg == "mean" else s_

            advantages[i] = adv - args.opd_kl_coef * cumulative_kl'''

LOSS_APPLY_REPLACEMENT = '''            else:
                masked_kl = reverse_kl

            # Form A-K: replace j>=1 sampled c_{t+j} with renormalized teacher top-K
            # reverse-KL d_K(h_{t+j}) (Rao-Blackwellized future). j=0 stays sampled.
            future_rb_enabled = (
                getattr(args, "opd_future_rb", False)
                and "teacher_topk_logp" in rollout_data
                and "student_topk_logp" in rollout_data
            )
            if future_rb_enabled:
                t_topk_logp = rollout_data["teacher_topk_logp"][i].to(device=device, dtype=torch.float32)
                s_topk_logp = rollout_data["student_topk_logp"][i].to(device=device, dtype=torch.float32)
                renorm = not getattr(args, "opd_future_no_renorm", False)
                d_k = _compute_renorm_topk_reverse_kl(s_topk_logp, t_topk_logp, renorm=renorm)
                masked_kl_future = d_k                 # j>=1 contributions (RB)
                masked_kl_current = masked_kl          # j=0 contribution (sampled, possibly IS-masked)
            else:
                masked_kl_future = masked_kl
                masked_kl_current = masked_kl

            if horizon <= 0 or horizon >= T:
                # full sequence (v2 semantics: never divide; agg only matters for truncated)
                if future_rb_enabled:
                    cumulative_kl = torch.zeros_like(reverse_kl)
                    if gamma == 1.0:
                        # Suffix sum of d_K, then swap t-th term to c_t.
                        suffix_d = torch.flip(torch.cumsum(torch.flip(masked_kl_future, [0]), dim=0), [0])
                        cumulative_kl = suffix_d + (masked_kl_current - masked_kl_future)
                    else:
                        running = torch.zeros((), dtype=reverse_kl.dtype, device=reverse_kl.device)
                        for t in range(T - 1, -1, -1):
                            cumulative_kl[t] = masked_kl_current[t] + gamma * running
                            running = masked_kl_future[t] + gamma * running
                elif gamma == 1.0:
                    cumulative_kl = torch.flip(torch.cumsum(torch.flip(masked_kl, [0]), dim=0), [0])
                else:
                    cumulative_kl = torch.zeros_like(reverse_kl)
                    running = torch.zeros((), dtype=reverse_kl.dtype, device=reverse_kl.device)
                    for t in range(T - 1, -1, -1):
                        running = masked_kl[t] + gamma * running
                        cumulative_kl[t] = running
            else:
                # truncated horizon: each token only looks K steps ahead
                cumulative_kl = torch.zeros_like(reverse_kl)
                for t in range(T):
                    end = min(t + horizon, T)
                    actual_k = end - t
                    if future_rb_enabled:
                        # j=0 sampled c_t; j>=1 RB d_K(h_{t+j}).
                        if gamma == 1.0:
                            s_ = masked_kl_current[t] + masked_kl_future[t + 1 : end].sum()
                        else:
                            s_ = masked_kl_current[t]
                            for j in range(1, actual_k):
                                s_ = s_ + (gamma ** j) * masked_kl_future[t + j]
                    elif gamma == 1.0:
                        s_ = masked_kl[t:end].sum()
                    else:
                        weights = gamma ** torch.arange(actual_k, device=reverse_kl.device, dtype=reverse_kl.dtype)
                        s_ = (masked_kl[t:end] * weights).sum()
                    cumulative_kl[t] = (s_ / actual_k) if agg == "mean" else s_

            advantages[i] = adv - args.opd_kl_coef * cumulative_kl'''

patch(
    path="/root/slime/slime/backends/megatron_utils/loss.py",
    marker="future_rb_enabled",
    locator=LOSS_APPLY_LOCATOR,
    replacement=LOSS_APPLY_REPLACEMENT,
    name="loss.py: apply_opd_kl_to_advantages add Form A-K future RB branch",
)


# ============================================================================
# data.py — add Form A-K keys to skip list in log_rollout_data
# ============================================================================
DATA_LOCATOR = '''                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
            ]:
                continue'''
DATA_REPLACEMENT = '''                "rollout_routed_experts",
                "max_seq_lens",
                "dynamic_global_batch_size",
                # Form A-K: 2-D per-position top-K data (shape [T, K]); not loggable as scalars.
                "teacher_topk_ids",
                "teacher_topk_logp",
                "student_topk_logp",
            ]:
                continue'''
patch(
    path="/root/slime/slime/backends/megatron_utils/data.py",
    marker='"teacher_topk_ids"',
    locator=DATA_LOCATOR,
    replacement=DATA_REPLACEMENT,
    name="data.py: skip list for Form A-K 2-D fields",
)


# ============================================================================
# actor.py — GPU-convert teacher_topk_ids/logp during data preprocessing
# ============================================================================
ACTOR_LOCATOR = '''        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:'''
ACTOR_REPLACEMENT = '''        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                torch.tensor(
                    slice_log_prob_with_cp(
                        log_prob,
                        total_length,
                        response_length,
                        self.args.qkv_format,
                        rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                    ),
                    device=torch.cuda.current_device(),
                    dtype=torch.float32,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        # Form A-K: per-position top-K teacher logprobs + ids. Each per-sample value is
        # list[response_length] of list[K]. CP slicer concatenates rows along time axis;
        # each row\'s K dimension is preserved.
        for key, dtype in (("teacher_topk_ids", torch.long), ("teacher_topk_logp", torch.float32)):
            if key not in rollout_data:
                continue
            sliced = []
            for i, (val, total_length, response_length) in enumerate(
                zip(
                    rollout_data[key],
                    rollout_data["total_lengths"],
                    rollout_data["response_lengths"],
                    strict=False,
                )
            ):
                local = slice_log_prob_with_cp(
                    val,
                    total_length,
                    response_length,
                    self.args.qkv_format,
                    rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                )
                sliced.append(torch.tensor(local, device=torch.cuda.current_device(), dtype=dtype))
            rollout_data[key] = sliced
        if "rollout_routed_experts" in rollout_data:'''
patch(
    path="/root/slime/slime/backends/megatron_utils/actor.py",
    marker='"teacher_topk_ids", torch.long',
    locator=ACTOR_LOCATOR,
    replacement=ACTOR_REPLACEMENT,
    name="actor.py: GPU-convert teacher_topk fields",
)


# ============================================================================
# model.py — pass teacher_topk_ids through forward_step for student log_probs pass
# ============================================================================
MODEL_IMPORT_LOC = "from .loss import loss_function"
MODEL_IMPORT_REP = "from .loss import get_log_probs_and_entropy, loss_function"
patch(
    path="/root/slime/slime/backends/megatron_utils/model.py",
    marker="from .loss import get_log_probs_and_entropy",
    locator=MODEL_IMPORT_LOC,
    replacement=MODEL_IMPORT_REP,
    name="model.py: import get_log_probs_and_entropy",
)

MODEL_FORWARD_LOC = '''        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
        }
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])
        output_tensor = model(**forward_kwargs)

        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )'''
MODEL_FORWARD_REP = '''        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "loss_masks",
                "multimodal_train_inputs",
                "total_lengths",
                "response_lengths",
                "max_seq_lens",
                "teacher_topk_ids",
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
        }
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])
        output_tensor = model(**forward_kwargs)

        # Form A-K: only forward teacher_topk_ids during the **student** log-prob pass
        # (no store_prefix). For ref/teacher passes (store_prefix='ref_'/'teacher_'),
        # we don\'t need student_topk_logp, so we pass None to skip the K-column CE.
        # Also, get_values() does not accept teacher_topk_ids.
        partial_kwargs = dict(
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )
        if (
            f is get_log_probs_and_entropy
            and store_prefix == ""
            and getattr(args, "opd_future_rb", False)
        ):
            partial_kwargs["teacher_topk_ids"] = batch.get("teacher_topk_ids")

        return output_tensor, partial(f, **partial_kwargs)'''
patch(
    path="/root/slime/slime/backends/megatron_utils/model.py",
    marker='partial_kwargs["teacher_topk_ids"]',
    locator=MODEL_FORWARD_LOC,
    replacement=MODEL_FORWARD_REP,
    name="model.py: extend forward_step for teacher_topk_ids",
)


# ============================================================================
# ray/rollout.py — serialize teacher_topk_ids/logp into train_data
# ============================================================================
ROLLOUT_TRAIN_DATA_LOC = '''        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        return train_data'''
ROLLOUT_TRAIN_DATA_REP = '''        if samples[0].teacher_log_probs is not None:
            train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

        # Form A-K: teacher top-K logprobs + token ids per response position.
        if samples[0].teacher_topk_ids is not None:
            train_data["teacher_topk_ids"] = [sample.teacher_topk_ids for sample in samples]
            train_data["teacher_topk_logp"] = [sample.teacher_topk_logp for sample in samples]

        return train_data'''
patch(
    path="/root/slime/slime/ray/rollout.py",
    marker='train_data["teacher_topk_ids"]',
    locator=ROLLOUT_TRAIN_DATA_LOC,
    replacement=ROLLOUT_TRAIN_DATA_REP,
    name="rollout.py: serialize teacher_topk_* into train_data",
)
ROLLOUT_KEYS_LOC = '''                "teacher_log_probs",
            ]:'''
ROLLOUT_KEYS_REP = '''                "teacher_log_probs",
                "teacher_topk_ids",
                "teacher_topk_logp",
            ]:'''
patch(
    path="/root/slime/slime/ray/rollout.py",
    marker='"teacher_topk_ids",\n                "teacher_topk_logp",',
    locator=ROLLOUT_KEYS_LOC,
    replacement=ROLLOUT_KEYS_REP,
    name="rollout.py: include teacher_topk_* in DP split keys",
)


# ============================================================================
# Apply
# ============================================================================
print(f"=== applying {len(PATCHES)} patches ===")
n_ok = 0
n_skip = 0
n_fail = 0
for p in PATCHES:
    path = Path(p["path"])
    if not path.exists():
        print(f"FAIL: {p['name']}: file does not exist: {path}")
        n_fail += 1
        continue
    src = path.read_text()
    if p["marker"] in src:
        print(f"SKIP: {p['name']}: marker already present")
        n_skip += 1
        continue
    if p["locator"] == "<UNUSED-WHOLE-FILE-REPLACEMENT>":
        path.write_text(p["replacement"])
        print(f"OK:   {p['name']}")
        n_ok += 1
        continue
    if p["locator"] not in src:
        print(f"FAIL: {p['name']}: locator string not found")
        n_fail += 1
        continue
    new_src = src.replace(p["locator"], p["replacement"], 1)
    path.write_text(new_src)
    print(f"OK:   {p['name']}")
    n_ok += 1

print()
print(f"Summary: {n_ok} applied, {n_skip} skipped (already), {n_fail} failed")
sys.exit(0 if n_fail == 0 else 1)
