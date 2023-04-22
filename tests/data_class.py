from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    train: int
    test: int
    num_classes: int = -1  # not provided

    @property
    def total(self):
        return self.train + self.test
