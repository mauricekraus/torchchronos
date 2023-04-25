from torchchronos.utils import parse_ts
from pathlib import Path


def test_parse_ucr_phoneme_ts():
    tsinfo = parse_ts(Path("./tests/fixtures/PhonemeSpectra_TEST.ts"))
    assert tsinfo.num_classes == 39
    assert tsinfo.series_length == 217
    assert tsinfo.equal_length == True
    assert tsinfo.dimensions == 11
    assert tsinfo.univariate == False
    assert tsinfo.missing == False


def test_parse_ucr_gunshot_ts():
    tsinfo = parse_ts(Path("./tests/fixtures/GunPoint_TEST.ts"))
    assert tsinfo.num_classes == 2
    assert tsinfo.series_length == 150
    assert tsinfo.equal_length == True
    assert tsinfo.dimensions == 1
    assert tsinfo.univariate == True
    assert tsinfo.missing == False


def test_parse_ucr_custom_missing():
    tsinfo = parse_ts(Path("./tests/fixtures/Custom_Missing_TEST.ts"))
    assert tsinfo.num_classes == 39
    assert tsinfo.series_length == 217
    assert tsinfo.equal_length == True
    assert tsinfo.dimensions == 11
    assert tsinfo.univariate == True
    assert tsinfo.missing == True
