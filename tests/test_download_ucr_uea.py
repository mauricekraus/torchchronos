# Pytest of download file
import os
from torchchronos.download import download_uea_ucr


def make_ucr_test(ds: str, path: str | None = ".cache/data", num_elems: int = 7):
    os.system(f"rm -rf .cache/data/{ds}")
    assert False == download_uea_ucr(ds, path)
    # assert that files where downloaded
    elems = os.listdir(f".cache/data/{ds}")
    endings = ["arff", "ts", "txt"]
    splits = ["TRAIN", "TEST"]
    assert len(elems) == num_elems
    for s, e in zip(splits, endings):
        assert f"{ds}_{s}.{e}" in elems

    assert True == download_uea_ucr(ds, path)


def test_download_ucr():
    make_ucr_test("GunPoint", None)
    make_ucr_test("Wafer")
    make_ucr_test("AbnormalHeartbeat", num_elems=8)
