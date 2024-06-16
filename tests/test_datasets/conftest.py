import pytest


def pytest_addoption(parser):
    parser.addoption("--download_all", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if config.getoption("--download_all"):
        # Don't skip tests if the special parameter is provided
        return

    skip_special = pytest.mark.skip(reason="need --download_all option to run")
    for item in items:
        if "download_all" in item.keywords:
            item.add_marker(skip_special)
