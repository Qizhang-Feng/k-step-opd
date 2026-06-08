import json

with open("/root/.cache/huggingface/Qwen3-4B-Base/tokenizer_config.json") as f:
    base_tok = json.load(f)
with open("/root/.cache/huggingface/sft-qwen3-4b-base-merged/tokenizer_config.json") as f:
    merged_tok = json.load(f)

print("=== Key token comparison ===")
print("eos_token: base=%s  merged=%s" % (base_tok.get("eos_token"), merged_tok.get("eos_token")))
print("pad_token: base=%s  merged=%s" % (base_tok.get("pad_token"), merged_tok.get("pad_token")))

base_ct = base_tok.get("chat_template", "")
merged_ct = merged_tok.get("chat_template", "")
print("chat_template same:", base_ct == merged_ct)
print("base chat_template len:", len(base_ct))
print("merged chat_template len:", len(merged_ct))

# Check added tokens
base_added = base_tok.get("added_tokens_decoder", {})
print("\nTotal added tokens:", len(base_added))
for tid, info in base_added.items():
    content = info.get("content", "")
    if "think" in content.lower():
        print("  Token %s: '%s' (special=%s)" % (tid, content, info.get("special")))

# Now check with transformers
from transformers import AutoTokenizer
print("\n=== Tokenizer loading test ===")
tok = AutoTokenizer.from_pretrained("/root/.cache/huggingface/sft-qwen3-4b-base-merged")
test = "<|im_start|>assistant\n<think>\n"
ids = tok.encode(test, add_special_tokens=False)
print("Encoded '%s' -> %s" % (test.replace("\n","\\n"), ids))
print("Decoded back:", repr(tok.decode(ids)))

# Check eos token id
print("\neos_token_id:", tok.eos_token_id)
print("eos_token:", repr(tok.eos_token))

# Check if </think> is a single token
think_end = tok.encode("</think>", add_special_tokens=False)
print("</think> token ids:", think_end)
