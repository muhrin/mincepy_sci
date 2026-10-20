"""Check that the types defined in the plugin modules are provided to mincePy"""

import importlib
import pathlib

import mincepy.plugins
import pytest

from mincepy_sci import provides

MODULES = sorted(path.stem for path in pathlib.Path(provides.__file__).parent.glob("*_types.py"))


@pytest.mark.parametrize("module_name", MODULES)
def test_module_types_are_provided(module_name):
    """Every module that defines types should have them provided by the plugin, otherwise mincePy
    can't use them"""
    try:
        module = importlib.import_module(f"mincepy_sci.{module_name}")
    except ImportError:
        pytest.skip(f"the library for {module_name} is not installed")

    module_types = getattr(module, "TYPES", ())
    if not module_types:
        pytest.skip(f"{module_name} doesn't define any types (yet)")

    provided = provides.get_types()
    missing = [helper for helper in module_types if helper not in provided]
    assert not missing, f"Types from {module_name} are not provided: {missing}"


def test_plugin_is_registered():
    """mincePy finds the types through the plugin entry point"""
    assert set(provides.get_types()) <= set(mincepy.plugins.get_types())


def test_type_ids_are_unique():
    type_ids = [helper.TYPE_ID for helper in provides.get_types()]
    assert len(type_ids) == len(set(type_ids))


# Types that are saved as part of others but that users don't use directly
INTERNAL_TYPES = {"AsArrayContext"}


def test_readme_lists_the_supported_types():
    """The README's list of supported types should be kept up to date"""
    readme = (pathlib.Path(provides.__file__).parents[2] / "README.rst").read_text()
    missing = []
    for helper in provides.get_types():
        # The first type is the public one (e.g. jax.Array rather than its implementation)
        obj_type = helper.TYPE[0] if isinstance(helper.TYPE, tuple) else helper.TYPE
        # Some compiled types have dotted names (e.g. '_jax.Array')
        name = obj_type.__name__.rsplit(".", 1)[-1]
        if name.startswith("_") or name in INTERNAL_TYPES:
            continue
        if f"``{name}``" not in readme:
            missing.append(f"{obj_type.__module__}.{name}")

    assert not missing, f"Supported types missing from the README: {missing}"
