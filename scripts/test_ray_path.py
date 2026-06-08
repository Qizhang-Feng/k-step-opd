import os
p = "/root/.cache/huggingface/sft-100k-merged"
print(f"path={p}")
print(f"exists={os.path.exists(p)}")
print(f"isdir={os.path.isdir(p)}")
if os.path.isdir(p):
    print(f"files={os.listdir(p)[:3]}")
else:
    print("NOT VISIBLE!")
    # Check parent
    parent = os.path.dirname(p)
    print(f"parent={parent} exists={os.path.exists(parent)}")
    if os.path.exists(parent):
        print(f"parent contents={os.listdir(parent)[:5]}")
