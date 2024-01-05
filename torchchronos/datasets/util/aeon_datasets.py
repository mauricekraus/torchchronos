import numpy as np
from prepareable_dataset import PrepareableDataset

"""
Classes for loading datasets from Aeon.
Regression, Classification classes needed.
Use PretrainDataset as a template.
"""


class AeonClassificationDataset(PrepareableDataset):
    def __init__(
        self,
        split: None | str = None,
        prepare: bool = False,
        load: bool = False,
        return_labels: bool = True,
    ) -> None:
        super().__init__(prepare, load)

        self.split = split
        self.return_labels = return_labels
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def _
