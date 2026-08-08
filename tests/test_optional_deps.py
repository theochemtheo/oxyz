"""The optional targets' import guards, and the tier environment they run in.

`oxyz.ase`, `oxyz.metatomic` and `oxyz.torch_sim` each wrap their third-party
imports in a `try`/`except ImportError` that re-raises a plain `ImportError`
naming the extra to install. That path only runs when the dependency is
genuinely absent, so in a full development environment it never executes and its
promise — that the message names the module and the extra, and chains the
original cause — goes untested.

These tests block the imports *in-process* rather than in a subprocess, so
coverage.py attributes those lines to the run that executes them. That keeps the
guards inside the full-dependency coverage measurement instead of only inside a
CI dependency tier, and it means the tests pass whether or not the optional
dependency happens to be installed.

The tier canary at the foot of the file checks the environment is the one CI
declared, so a tier that quietly installed too much — or a skip guard that fires
for the wrong reason — surfaces as a failure rather than a green run over
nothing.
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

    Raises rather than returning `None`: returning `None` means "not my
    business, ask the next finder", which would fall through to the real one.
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
    # A cached module never re-executes, so the guard would not run. Drop the
    # target and anything under the blocked roots; the rest of oxyz stays cached.
    saved_modules = {name: sys.modules[name] for name in touched()}
    for name in saved_modules:
        del sys.modules[name]
    sys.meta_path.insert(0, _Blocker(roots))
    try:
        yield
    finally:
        sys.meta_path[:] = saved_meta
        # Restore only these names. Restoring a whole sys.modules snapshot would
        # also un-import whatever the blocked import pulled in on its way to the
        # guard — and if that included `oxyz._rust`, the next import of it fails
        # with "cannot load module more than once per process", because a PyO3
        # abi3 extension cannot be initialised twice. Anything newly imported is
        # harmless to leave in place.
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
    # Exactly ImportError, not a ModuleNotFoundError: the latter would mean the
    # raw import error escaped and the guard never ran.
    assert type(error) is ImportError
    assert module in str(error)
    assert extra in str(error)
    # `raise ... from error` keeps the underlying failure reachable, so a typo in
    # an internal import is still diagnosable behind the friendly message.
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


# The dependency tiers CI builds, as the optional top-level packages each is
# expected to provide. `OXYZ_TEST_TIER` is set by the `tiers`, `python` and
# `coverage` jobs in test.yml; unset (a bare local run) skips, because there is
# no declared expectation to check against.
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
