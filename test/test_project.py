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


def _python_code_blocks(rst_file: pathlib.Path) -> list[str]:
    """Get the code from the '.. code-block:: python' blocks of a reStructuredText file"""
    lines = rst_file.read_text().splitlines()
    blocks = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() == ".. code-block:: python":
            idx += 1
            block = []
            while idx < len(lines) and (not lines[idx].strip() or lines[idx].startswith("    ")):
                block.append(lines[idx][4:])
                idx += 1
            blocks.append("\n".join(block))
        else:
            idx += 1
    return blocks


def test_readme_example_runs(tmp_path):
    """The quick start example in the README should work"""
    for module in ("numpy", "pymatgen.core", "torch"):
        pytest.importorskip(module)

    (code,) = _python_code_blocks(ROOT / "README.rst")
    code = code.replace("/path/to/my/db", str(tmp_path / "db"))
    code += """

assert len(energies) == 7
assert nacl.composition.reduced_formula == "NaCl"
assert [type(obj).__name__ for obj in structures] == ["Structure"]
"""
    exec(compile(code, "README.rst", "exec"), {})  # nosec
