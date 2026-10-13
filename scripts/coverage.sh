#!/usr/bin/env bash
# Python and Rust coverage from one run: the extension is instrumented, so
# pytest also covers the Rust it reaches. Expects `uv sync --extra ase`.
set -euo pipefail

PY_MIN="${COVERAGE_PY_MIN:-97}"
RS_MIN="${COVERAGE_RS_MIN:-90}"

# The profiler runtime needs rustup's cargo, not a system one.
if command -v rustup >/dev/null; then
    PATH="$(dirname "$(rustup which cargo)"):$PATH"
fi

mkdir -p coverage

cov_env="$(cargo llvm-cov show-env --sh 2>/dev/null)"
cov_vars="$(printf '%s\n' "$cov_env" | sed -n 's/^export \([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p')"
eval "$cov_env"

restore_build() {
    # Put back the release build; CI discards the environment anyway.
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
