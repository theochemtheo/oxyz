"""Every example script runs cleanly, and the scoreboard matches the tree."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

# Scoreboard: every runnable example, listed explicitly. Adding a script to
# examples/ without adding it here (or vice versa) fails test_scoreboard_matches.
EXAMPLES: tuple[str, ...] = (
    "scan_and_schema.py",
    "read_numpy.py",
    "batch_training_loop.py",
    "schema_and_project.py",
    "ase_dropin.py",
    "pytorch_targets.py",
    "write_roundtrip.py",
)


def test_scoreboard_matches_tree() -> None:
    on_disk = {p.name for p in EXAMPLES_DIR.glob("*.py")}
    assert on_disk == set(EXAMPLES)


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_runs(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / name)],
        capture_output=True,
        text=True,
        cwd=EXAMPLES_DIR.parent,
    )
    assert result.returncode == 0, result.stderr
