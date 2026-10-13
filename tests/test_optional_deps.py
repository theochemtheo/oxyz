"""The optional targets' import guards, and the CI tier canary.

Imports are blocked in-process, so coverage sees the guards and the tests
pass whether or not the dependency is installed.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from importlib.machinery import ModuleSpec
    from types import ModuleType

# (module, top-level names its guard imports, the extra its message must name)
TARGETS = [
    ("oxyz.ase", ("ase",), "oxyz[ase]"),
    ("oxyz.metatomic", ("torch", "metatomic"), "oxyz[metatomic]"),
    ("oxyz.torch_sim", ("torch", "torch_sim"), "oxyz[torch-sim]"),
]


class _Blocker:
    """A meta-path finder that makes the named roots unimportable.

    It raises: returning `None` would fall through to the real finder.
    """

    def __init__(self, roots: tuple[str, ...]) -> None:
        self._roots = set(roots)

    def find_spec(
        self, fullname: str, path: object = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname.partition(".")[0] in self._roots:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


@contextlib.contextmanager
def _hidden(target: str, roots: tuple[str, ...]) -> Iterator[None]:
    """Make `roots` unimportable and force `target` to re-run its guard."""

    def touched() -> list[str]:
        return [
            name
            for name in sys.modules
            if name == target or name.partition(".")[0] in roots
        ]

    saved_meta = list(sys.meta_path)
    # A cached module never re-executes its guard.
    saved_modules = {name: sys.modules[name] for name in touched()}
    for name in saved_modules:
        del sys.modules[name]
    sys.meta_path.insert(0, _Blocker(roots))
    try:
        yield
    finally:
        sys.meta_path[:] = saved_meta
        # Restore only these: un-importing `oxyz._rust` would break it, as a PyO3
        # abi3 extension cannot be initialised twice.
        for name in touched():
            del sys.modules[name]
        sys.modules.update(saved_modules)


@pytest.mark.parametrize(
    ("module", "roots", "extra"), TARGETS, ids=[target[0] for target in TARGETS]
)
def test_missing_dependency_raises_with_the_install_hint(
    module: str, roots: tuple[str, ...], extra: str
) -> None:
    with _hidden(module, roots), pytest.raises(ImportError) as excinfo:
        importlib.import_module(module)

    error = excinfo.value
    # A ModuleNotFoundError would mean the guard never ran.
    assert type(error) is ImportError
    assert module in str(error)
    assert extra in str(error)
    assert error.__cause__ is not None


@pytest.mark.parametrize(
    ("module", "roots"),
    [(module, roots) for module, roots, _ in TARGETS],
    ids=[target[0] for target in TARGETS],
)
def test_hiding_a_dependency_leaves_no_trace(
    module: str, roots: tuple[str, ...]
) -> None:
    """The blocker must not leak: a later import behaves as it did before."""
    before = importlib.util.find_spec(roots[0]) is not None

    with _hidden(module, roots), contextlib.suppress(ImportError):
        importlib.import_module(module)

    assert (importlib.util.find_spec(roots[0]) is not None) is before
    assert not any(isinstance(finder, _Blocker) for finder in sys.meta_path)


# The optional packages each CI tier provides, keyed by `OXYZ_TEST_TIER`.
TIERS = {
    "base": set(),
    "ase": {"ase"},
    "s3": {"boto3", "moto", "obstore"},
    "full": {"ase", "boto3", "metatomic", "moto", "obstore", "torch", "torch_sim"},
}
_OPTIONAL = sorted(set().union(*TIERS.values()))


def test_tier_environment_is_the_one_ci_declared() -> None:
    tier = os.environ.get("OXYZ_TEST_TIER")
    if tier is None:
        pytest.skip("OXYZ_TEST_TIER unset; not a tiered run")

    assert tier in TIERS, f"unknown tier {tier!r}; expected one of {sorted(TIERS)}"
    present = {name for name in _OPTIONAL if importlib.util.find_spec(name) is not None}
    assert present == TIERS[tier], (
        f"tier {tier!r} resolved to an unexpected environment.\n"
        f"  missing: {sorted(TIERS[tier] - present)}\n"
        f"  unexpected: {sorted(present - TIERS[tier])}\n"
        "Either the install step drifted from the tier table in this file, or a "
        "dependency group changed what it pulls in."
    )
