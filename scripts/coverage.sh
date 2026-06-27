#!/usr/bin/env bash
# Measure test coverage for both languages and assemble a combined report.
#
# One instrumented pytest run feeds both reports: coverage.py measures the
# Python package (src/oxyz), and because the Rust extension is built with
# coverage instrumentation, the same run records which lines of oxyz-core and
# oxyz-py it executes through the binding. cargo test adds the Rust-only paths.
# Without this, oxyz-py (reached only from Python) would read as 0%.
#
# The two summaries are concatenated into coverage/summary.md under Python and
# Rust headings; the report is written before the floors are enforced, so a
# failing gate still leaves it behind.
#
# Assumes `uv sync --extra ase` has been run. Outputs: htmlcov/{python,rust}/
# (browse), coverage/{python,rust}.lcov, coverage/summary.md.
# Run from the repository root:  bash scripts/coverage.sh
set -euo pipefail

# Minimum line coverage, percent. Override per run via the environment.
PY_MIN="${COVERAGE_PY_MIN:-97}"
RS_MIN="${COVERAGE_RS_MIN:-90}"

# cargo-llvm-cov needs the rustup toolchain for the profiler runtime; ensure its
# cargo shadows any system one (e.g. MacPorts/Homebrew) on PATH.
if command -v rustup >/dev/null; then
    PATH="$(dirname "$(rustup which cargo)"):$PATH"
fi

mkdir -p coverage

# Coverage instrumentation env (RUSTC_WRAPPER, LLVM_PROFILE_FILE, ...). Capture
# the variable names too, so the normal build can be restored afterwards.
cov_env="$(cargo llvm-cov show-env --sh 2>/dev/null)"
cov_vars="$(printf '%s\n' "$cov_env" | sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p')"
eval "$cov_env"

restore_build() {
    # maturin develop installed an instrumented debug build into the venv;
    # leave the project on its normal release build. Skipped in CI, where the
    # environment is discarded anyway.
    [ -n "${CI:-}" ] && return 0
    unset $cov_vars
    uv sync --extra ase --reinstall-package oxyz >/dev/null
}
trap restore_build EXIT

cargo llvm-cov clean --workspace
cargo test --workspace
uv run --no-sync maturin develop
uv run --no-sync pytest --cov=oxyz --cov-report=html --cov-report=lcov --cov-report=term-missing
uv run --no-sync coverage report --format=markdown > coverage/python.md

cargo llvm-cov report --lcov --output-path coverage/rust.lcov
cargo llvm-cov report --html --output-dir htmlcov/rust
cargo llvm-cov report --summary-only > coverage/rust.summary.txt

{
    echo "## Coverage"
    echo
    echo "### Python"
    cat coverage/python.md
    echo
    echo "### Rust"
    echo '```'
    cat coverage/rust.summary.txt
    echo '```'
} > coverage/summary.md

uv run --no-sync coverage report --fail-under="$PY_MIN" > /dev/null
cargo llvm-cov report --fail-under-lines "$RS_MIN" > /dev/null
