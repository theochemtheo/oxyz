# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "oxyz",
#     "pytest>=9",
#     "pytest-benchmark>=5.2",
#     "ase>=3.28,<4",
#     "extxyz>=0.4",
#     "ase-extxyz>=0.1",
#     "atompack-db>=0.4",
#     "lmdb>=1.4",
#     "torch>=2",
#     "metatomic-torch",
#     # _style (palette/labels) imports seaborn; without it the style unit
#     # tests silently skip and stop guarding the figure labels.
#     "seaborn>=0.13",
# ]
#
# [tool.uv.sources]
# oxyz = { path = ".." }
# torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Run the benchmark suite in its own environment.

uv reads the inline metadata above and supplies everything the suite
needs: CPython 3.13 (cextxyz and ase-extxyz publish no 3.14 wheels), the
comparison libraries, and a release build of this checkout's oxyz
(rebuilt when the Rust sources change, via the cache-keys in
pyproject.toml). The project venv plays no part. Arguments pass through
to pytest:

    uv run benchmarks/run.py --benchmark-autosave    # record a run
    uv run benchmarks/run.py --benchmark-disable     # smoke test
    uv run benchmarks/report.py                      # render RESULTS.md
    uv run benchmarks/plot.py                        # render figures (bars + curves)

test_scaling.py adds size and thread sweeps recorded into the same save; the
scaling_* groups back the curve figures.
"""

import sys
from pathlib import Path

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).parent), *sys.argv[1:]]))
