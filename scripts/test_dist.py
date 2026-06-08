import os, time, torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

sizes_mb = [1, 128, 1024]
for mb in sizes_mb:
    x = torch.ones((mb * 1024 * 1024 // 2,), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = time.time() - t0
    if rank == 0:
        print(f"{mb}MB x20 all_reduce: {dt:.2f}s, {20*mb/dt:.1f} MB/s", flush=True)

dist.barrier()
dist.destroy_process_group()
