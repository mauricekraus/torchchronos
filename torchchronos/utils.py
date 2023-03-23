from pathlib import Path

import torch


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def swap_batch_seq_collate_fn(data):
    xs, ys = zip(*data)

    xs = torch.stack(xs).permute(1, 0, 2)

    return xs, torch.stack(ys)
