"""Smoke test: every module in the package imports cleanly."""
import importlib

import pytest

MODULES = [
    "src",
    "src.config",
    "src.data_loader",
    "src.preprocessing",
    "src.features",
    "src.dataset",
    "src.hmm",
    "src.training",
    "src.evaluation",
    "src.visualization",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)
