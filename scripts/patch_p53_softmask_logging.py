#!/usr/bin/env python3
"""Add soft-mask diagnostics logging to p5-3's loss.py.

Patches:
1. Per-rollout summary stats: soft_weight_mean, is_ratio_mean, is_ratio_max, frac_downweighted
2. KL dump: add rollout_log_probs per-sample for offline IS-ratio analysis

Run inside p5-3 k-step-opd container. Idempotent.
"""
from pathlib import Path
import shutil
import time
import sys


def patch_logging():
    p = Path("/root/slime/slime/backends/megatron_utils/loss.py")
    s = p.read_text()

    if "soft_weight_mean" in s:
        print("loss.py: soft_weight logging already present, skipping")
        return

    # 1. Add per-rollout soft_weight stats collection.
    # Insert after the soft_weight computation, inside the cumulative block.
    # We accumulate stats across samples, then store in rollout_data after the loop.

    # Find the line "reverse_kls.append(reverse_kl)" and add collection after each sample
    anchor_collect = "        reverse_kls.append(reverse_kl)"
    if anchor_collect not in s:
        print("ERROR: cannot find reverse_kls.append anchor")
        sys.exit(1)

    # Add a collector list before the for loop
    # Find "    reverse_kls = []"
    anchor_init = "    reverse_kls = []"
    if anchor_init not in s:
        print("ERROR: cannot find reverse_kls init")
        sys.exit(1)

    s = s.replace(
        anchor_init,
        anchor_init + "\n    _is_ratios_all = []  # collect IS ratios for logging\n    _soft_weights_all = []  # collect soft weights for logging",
        1
    )

    # After soft_weight is computed, append it; also append is_ratio
    # Find the soft_weight branch
    anchor_soft = "                    soft_weight = torch.clamp(dualclip_c / torch.clamp(is_ratio, min=1.0), max=1.0)"
    if anchor_soft not in s:
        print("ERROR: cannot find soft_weight line")
        sys.exit(1)

    s = s.replace(
        anchor_soft,
        anchor_soft + "\n                    _is_ratios_all.append(is_ratio.detach())\n                    _soft_weights_all.append(soft_weight.detach())",
        1
    )

    # Also for hard mask branch, collect is_ratio
    anchor_hard = "                    keep_mask = (is_ratio <= dualclip_c).to(reverse_kl.dtype)\n                    masked_kl = reverse_kl * keep_mask"
    if anchor_hard in s:
        s = s.replace(
            anchor_hard,
            anchor_hard + "\n                    _is_ratios_all.append(is_ratio.detach())",
            1
        )

    # After the for loop ends (after "rollout_data[\"opd_reverse_kl\"] = reverse_kls"),
    # add the summary stats to rollout_data
    anchor_store = '    rollout_data["opd_reverse_kl"] = reverse_kls'
    if anchor_store not in s:
        print("ERROR: cannot find opd_reverse_kl store line")
        sys.exit(1)

    stats_block = '''

    # Soft-mask / IS-ratio diagnostics (for paper analysis)
    if _is_ratios_all:
        _all_is = torch.cat(_is_ratios_all)
        rollout_data["_is_ratio_mean"] = float(_all_is.mean())
        rollout_data["_is_ratio_max"] = float(_all_is.max())
        rollout_data["_frac_downweighted"] = float((_all_is > dualclip_c).float().mean()) if dualclip_c > 0 else 0.0
    if _soft_weights_all:
        _all_sw = torch.cat(_soft_weights_all)
        rollout_data["_soft_weight_mean"] = float(_all_sw.mean())
        rollout_data["_soft_weight_min"] = float(_all_sw.min())
'''
    s = s.replace(anchor_store, anchor_store + stats_block, 1)

    # 2. Add rollout_log_probs to KL dump for offline analysis
    # Find where we build the dump record dict. Look for '"student_log_probs":' in the dump function.
    anchor_dump_record = '"reverse_kl": rkl,'
    if anchor_dump_record not in s:
        # Might be formatted differently
        anchor_dump_record = '"reverse_kl": rkl'

    if anchor_dump_record in s:
        # Add rollout_logp to the dump record
        # First need to get rollout_log_probs into the dump function
        # Check if rollout_log_probs is already passed to _dump_opd_kl
        if "rollout_log_probs" not in s.split("def _dump_opd_kl")[1].split("def ")[0][:200]:
            # Need to pass it. But this is complex — let's just grab it from rollout_data inside the dump func
            # Actually it's already available: rollout_data.get("rollout_log_probs") can be called inside _dump_opd_kl
            pass

        # Add rollout_logp to the jsonl record
        # Find the record dict construction
        if '"reverse_kl": rkl' in s:
            old_record_line = '"reverse_kl": rkl'
            # Check what comes after
            idx = s.index(old_record_line)
            # Add rollout_logp field
            new_field = '''
                "rollout_log_probs": rlp,'''
            # We need to also extract rollout_log_probs in the dump loop
            # Find where slp/tlp/rkl are extracted in the dump function
            anchor_extract = "            slp = student_log_probs[i].detach()"
            if anchor_extract in s:
                s = s.replace(
                    anchor_extract,
                    '            _rlp_list = rollout_data.get("rollout_log_probs", [])\n'
                    '            rlp = _rlp_list[i].detach().to(dtype=torch.float32, device="cpu").tolist() if i < len(_rlp_list) else []\n'
                    + anchor_extract,
                    1
                )
                # Add to the record dict
                s = s.replace(
                    '"reverse_kl": rkl,',
                    '"reverse_kl": rkl,\n                "rollout_log_probs": rlp,',
                    1
                )
            else:
                print("WARNING: could not add rollout_log_probs to dump (extract anchor not found)")
        else:
            print("WARNING: could not find reverse_kl record line for dump patch")
    else:
        print("WARNING: dump record anchor not found, skipping dump patch")

    backup = p.with_suffix(p.suffix + f".pre-logging-{int(time.time())}")
    shutil.copy(p, backup)
    p.write_text(s)
    print(f"loss.py: added soft-mask logging + rollout_log_probs to dump (backup: {backup.name})")


if __name__ == "__main__":
    patch_logging()
    print("DONE")
