"""Checks on the project itself: that the packages contain what they should and that the project
metadata is consistent"""

import pathlib
import tarfile
import tomllib
import zipfile

import mincepy
from packaging.requirements import Requirement
import pytest
import yaml

ROOT = pathlib.Path(__file__).parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _classifier_versions() -> list[str]:
    prefix = "Programming Language :: Python :: 3."
    return [
        classifier.rsplit(" :: ", 1)[1]
        for classifier in PYPROJECT["project"]["classifiers"]
        if classifier.startswith(prefix)
    ]


def test_python_versions_are_consistent():
    """The supported Python versions are given in several places, they should all agree"""
    versions = _classifier_versions()
    oldest = versions[0]
    assert versions == sorted(versions, key=lambda version: tuple(map(int, version.split("."))))

    assert PYPROJECT["project"]["requires-python"] == f">={oldest}"
    assert PYPROJECT["tool"]["ruff"]["target-version"] == "py" + oldest.replace(".", "")
    assert PYPROJECT["tool"]["ty"]["environment"]["python-version"] == oldest

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    tested = workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"]
    assert tested == versions


def test_installed_mincepy_is_supported():
    """The mincePy being tested against should be one that this package says it supports,
    otherwise e.g. 'pip install mincepy[sci]' won't pick up this version"""
    (requirement,) = (
        Requirement(dep)
        for dep in PYPROJECT["project"]["dependencies"]
        if dep.startswith("mincepy")
    )
    assert requirement.specifier.contains(mincepy.__version__, prereleases=True)


def test_packages_contain_the_sources(tmp_path):
    buildapi = pytest.importorskip("flit_core.buildapi")

    sdist = tmp_path / buildapi.build_sdist(str(tmp_path))
    with tarfile.open(sdist) as archive:
        names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    assert "src/mincepy_sci/__init__.py" in names
    assert "src/mincepy_sci/provides.py" in names
    assert "README.rst" in names

    wheel = tmp_path / buildapi.build_wheel(str(tmp_path))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "mincepy_sci/__init__.py" in names
    assert "mincepy_sci/provides.py" in names
    entry_points = next(name for name in names if name.endswith("entry_points.txt"))
    with zipfile.ZipFile(wheel) as archive:
        assert "mincepy_sci.provides:get_types" in archive.read(entry_points).decode()


@pytest.mark.parametrize("path", ["README.rst", "pyproject.toml"])
def test_no_stale_references(path):
    """References to services and tools the project no longer uses"""
    text = (ROOT / path).read_text().lower()
    for stale in ("travis", "setup.py", "code%20style-black", "prospector", "yapf"):
        assert stale not in text, f"'{stale}' found in {path}"
