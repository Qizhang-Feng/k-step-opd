"""Container-specific patch for model.py forward_step (forward_only path)."""
import sys

path = "/root/slime/slime/backends/megatron_utils/model.py"
src = open(path).read()

if 'partial_kwargs["teacher_topk_ids"]' in src:
    print("already patched")
    sys.exit(0)

# Container-specific anchor: the forward_step *inside* forward_only function.
locator = '''        # Get the batch.
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
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )

        return output_tensor, partial(
            f,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )'''

replacement = '''        # Get the batch.
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
        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["full_loss_masks"],
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
        )

        # Form A-K: pass teacher_topk_ids through to get_log_probs_and_entropy ONLY
        # during the student log-prob pass (no store_prefix). Ref/teacher passes don't
        # need student_topk_logp; get_values() doesn't accept teacher_topk_ids.
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

if locator not in src:
    print("FAIL: locator string not found in file")
    # Print a few candidate lines for debugging
    sys.exit(1)

new_src = src.replace(locator, replacement, 1)
open(path, "w").write(new_src)
print("OK: model.py forward_step patched")
