from pathlib import Path

import torch
from dataclasses import dataclass
from pathlib import Path


def swap_batch_seq_collate_fn(data):
    xs, ys = zip(*data)

    xs = torch.stack(xs).permute(1, 0, 2)

    return xs, torch.stack(ys)


@dataclass
class TSInfo:
    num_classes: int
    series_length: int
    equal_length: bool
    dimensions: int
    univariate: bool


def parse_ts(path: Path):
    num_classes = 0
    series_length = -1
    equal_length = False
    dimensions = -1
    univariate = False

    with path.open("r") as f:
        for line in f:
            if "@data" in line:
                break

            split = line.strip().split(" ")
            if split[0] == "@classLabel":
                if split[1] == "true":
                    num_classes = len(split[2:])
            elif split[0] == "@seriesLength" and split[1].isdigit():
                series_length = int(split[1])
            elif split[0] == "@equalLength" and split[1] == "true":
                equal_length = True
            elif split[0] == "@dimensions" and split[1].isdigit():
                dimensions = int(split[1])
            elif split[0] == "@univariate" and split[1] == "true":
                univariate = True
                dimensions = 1
    return TSInfo(num_classes, series_length, equal_length, dimensions, univariate)
