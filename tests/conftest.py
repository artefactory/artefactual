import importlib.util

import pytest

VLLM_INSTALLED = importlib.util.find_spec("vllm") is not None


def pytest_configure(config):
    config.addinivalue_line("markers", "vllm: mark test as requiring vllm")


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    if VLLM_INSTALLED:
        return
    skip_vllm = pytest.mark.skip(reason="vllm not installed")
    for item in items:
        if "vllm" in item.keywords:
            item.add_marker(skip_vllm)
