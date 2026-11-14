# Writing documentation

Public Markdown lives under `docs/`; the README is the repository introduction.
Keep guides task-focused, concepts explanatory and reference pages useful for
lookup. Use British spelling and relative `.md` links between public pages.
Link to repository source with absolute GitHub URLs so links also work in the
rendered site. Do not link to private notes or publish scratch material.

## Run examples

From the checkout root, with the full development environment:

```sh
uv sync --extra ase
uv run pytest docs src/oxyz
uv run pytest -q tests/test_doc_examples.py
```

Sybil checks `python` fences in public Markdown and doctests in package
source docstrings. Root `README.md` has a collector route, but its conversion
and inclusion in default collection are pending; do not treat its existing
fragments as verified examples. Plans, skills, incidental strings in tests and
`docs/superpowers/` are not example sources.

Each document gets its own namespace and fresh temporary working directory.
Blocks within a page share names and files in source order; pages cannot depend
on another page. The original working directory is restored even after a failed
example. Ordinary tests do not use this fixture. Existing source docstrings
retain a compatibility `DATA` path to `examples/data`; new Markdown must show
all imports and input setup explicitly, without hidden fixtures or bindings.

## Write a complete example

Use a plain `python` fence for workflows. Assert the values that matter rather
than merely showing that a call returns. This example creates its input and
checks a read/write round trip without depending on the checkout directory:

```python
from pathlib import Path

import oxyz

source = Path("one-atom.extxyz")
source.write_text("1\nProperties=species:S:1:pos:R:3 energy=-1.5\nH 0.25 0 0\n")
frame = oxyz.read(source, 0)
assert frame.n_atoms == 1
assert frame.columns["species"] == ["H"]
assert frame.columns["pos"][0, 0] == 0.25
assert frame.metadata["energy"] == -1.5

oxyz.write("roundtrip.extxyz", frame)
restored = oxyz.read("roundtrip.extxyz", 0)
assert restored.n_atoms == 1
assert restored.columns["pos"].tolist() == [[0.25, 0.0, 0.0]]
assert restored.metadata["energy"] == -1.5
```

Use a doctest-style `python` fence when displayed output helps the reader.
Expected output is checked, not just executed:

```python
>>> restored.n_atoms
1
>>> restored.metadata["energy"]
-1.5
```

The configured doctest flags are `NORMALIZE_WHITESPACE | ELLIPSIS`. Prefer short,
exact output to full object representations. Avoid timestamps, random samples,
absolute paths, timings, nondeterministic ordering and device-dependent output.
There is no global numeric tolerance: for rounded floating-point calculations,
use `numpy.testing.assert_allclose` with explicit `rtol` and `atol` appropriate
to the example. Use assertions for array shapes, selected values and identities.

## Schematic and optional examples

Runnable examples must not require network access, credentials or large external
corpora. A genuine schematic example may use a named-reason Sybil skip comment:

<!-- skip: next "schematic URL requires private remote credentials and data" -->

```python
frames = oxyz.read("s3://private-bucket/trajectory.extxyz")
```

The syntax is `<!-- skip: next "reason" -->` immediately before the example.
Use a reason describing the unavailable input, not a blanket skip for a failing
example. Reported skips are visible with `pytest -rs`. Existing mocked remote
tests establish remote behaviour; a skipped URL does not prove a network read.

Optional converter examples run in the full development environment and full CI.
Missing extras there are failures. Isolated wheel tiers explicitly collect only
`tests`, not the documentation tree; do not add environment-dependent blanket
skips to the collector.

Shell fences (`sh`, `bash`) document commands but are not executed by Sybil.
Exercise CLI behaviour through the existing CLI tests and an explicit local
smoke command. Do not claim shell-fence coverage.
