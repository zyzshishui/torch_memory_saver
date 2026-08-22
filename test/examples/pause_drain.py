import sys

import torch
from torch_memory_saver import torch_memory_saver


def run(hook_mode: str):
    torch_memory_saver.hook_mode = hook_mode

    element_count = 8 * 1024 * 1024
    with torch_memory_saver.region(tag="inflight"):
        buffer = torch.ones(element_count, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()

    side_stream = torch.cuda.Stream()
    with torch.cuda.stream(side_stream):
        torch.cuda._sleep(1_000_000_000)

    torch_memory_saver.pause("inflight")

    assert side_stream.query(), "pause() returned before in-flight device work completed"
    torch_memory_saver.resume("inflight")
    del buffer


if __name__ == "__main__":
    run(hook_mode=sys.argv[1])
