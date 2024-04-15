from pathlib import Path

__author__ = "The torchchronos contributors"
__version__ = "0.1.0"


cache_path = Path.home() / ".torchchronos"
cache_path.mkdir(exist_ok=True)

dataset_cache_path = cache_path / "datasets"
dataset_cache_path.mkdir(exist_ok=True)
