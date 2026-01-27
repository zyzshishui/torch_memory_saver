"""Not to be used by end users, but only for tests of the package itself."""

import torch


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message."""
    if torch.version.hip:
        # ROCm: amd-smi (device_memory_used) has delays, use mem_get_info for real-time tracking
        free, total = torch.cuda.mem_get_info(gpu_id)
        mem = total - free
    else:
        mem = torch.cuda.device_memory_used(gpu_id)
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem
