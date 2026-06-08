import asyncio

import aiohttp
import torch

from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    # Retry on transient teacher errors (500, timeouts, connection resets).
    # Critically, we must NOT let aiohttp.ClientResponseError propagate to Ray:
    # it carries a CIMultiDictProxy (headers) that is not picklable and crashes
    # the whole job. We convert any failure into a plain RuntimeError.
    max_retries = 5
    last_err = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(args.rm_url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"teacher returned status {resp.status}: {body[:200]}")
                    return await resp.json()
        except Exception as e:  # noqa: BLE001
            # Strip any unpicklable payload by keeping only the message string.
            last_err = RuntimeError(
                f"teacher request failed (attempt {attempt + 1}/{max_retries}): {str(e)[:300]}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(2.0 * (attempt + 1))
            continue
    raise last_err


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    # Extract teacher log-probs from the sglang response
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

    # Return scalar rewards for GRPO/PPO advantage estimator (pure distillation -> 0.0).
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
