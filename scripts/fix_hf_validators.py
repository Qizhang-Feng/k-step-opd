"""Patch huggingface_hub _validators.py to allow local paths."""
import os
import sys

VFILE = "/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_validators.py"

with open(VFILE) as f:
    content = f.read()

# Check if already patched
if "_os.path.exists" in content:
    print("Already patched!")
    sys.exit(0)

old = "def validate_repo_id(repo_id: str | None) -> None:"
new = """def validate_repo_id(repo_id: str | None) -> None:
    import os as _os
    if isinstance(repo_id, str) and _os.path.exists(repo_id):
        return"""

if old not in content:
    # Try alternative signature
    old = "def validate_repo_id(repo_id: str) -> None:"
    new = """def validate_repo_id(repo_id: str) -> None:
    import os as _os
    if isinstance(repo_id, str) and _os.path.exists(repo_id):
        return"""

if old not in content:
    print(f"ERROR: Could not find function signature in {VFILE}")
    # Print first few lines of the function
    for line in content.split("\n"):
        if "validate_repo_id" in line:
            print(f"  Found: {line}")
    sys.exit(1)

content = content.replace(old, new, 1)
with open(VFILE, "w") as f:
    f.write(content)

# Clear pycache
import shutil
cache_dir = os.path.dirname(VFILE) + "/__pycache__"
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

print("Patched successfully!")

# Verify
from transformers import AutoConfig
c = AutoConfig.from_pretrained("/root/.cache/huggingface/sft-100k-merged", trust_remote_code=True)
print(f"Verified: {c.model_type}")
