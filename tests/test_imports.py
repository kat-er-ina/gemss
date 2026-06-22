"""Tests that every gemss module can be imported without error."""

import importlib
import pkgutil

import pytest

import gemss


def _all_modules() -> list[str]:
    modules = ['gemss']
    for info in pkgutil.walk_packages(gemss.__path__, prefix='gemss.'):
        modules.append(info.name)
    return modules


@pytest.mark.parametrize('module', _all_modules())
def test_module_importable(module: str) -> None:
    importlib.import_module(module)
