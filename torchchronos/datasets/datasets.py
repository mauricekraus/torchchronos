from .util.aeon_datasets import AeonDataset


class EMGDataset(AeonDataset):
    def __init__(
        self,
        split: str | None = None,
        save_path: str | None = None,
        prepare: bool = False,
        load: bool = False,
        has_y: bool = True,
        return_labels: bool = True,
        use_cache: bool = True
    ) -> None:
        super().__init__("NerveDamage", split, save_path, prepare, load, has_y, return_labels, use_cache)



