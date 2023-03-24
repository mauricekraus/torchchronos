# Pytest of download file
import os
from torchchronos.download import download_uea_ucr


def test_download_ucr():
    os.system("rm -rf .cache/data/GunPoint")
    download_uea_ucr(None, "GunPoint")
    # assert that files where downloaded
    elems = os.listdir(".cache/data/GunPoint")
    endings = ["arff", "ts", "txt"]
    splits = ["TRAIN", "TEST"]
    assert len(elems) == 7
    for s, e in zip(splits, endings):
        assert f"GunPoint_{s}.{e}" in elems
