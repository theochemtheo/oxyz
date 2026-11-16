# Writing documentation

Public Markdown lives under `docs/`; the README is the repository introduction.
Keep guides task-focused, concepts explanatory and reference pages useful for
lookup. Use British spelling and relative `.md` links between public pages.
Link to repository source with absolute GitHub URLs so links also work in the
rendered site. Do not link to private notes or publish scratch material.

## Run examples

From the checkout root, with the full development environment:

```sh title="Check documentation from the checkout"
uv sync --extra ase
uv run pytest docs README.md src/oxyz
uv run pytest -q tests/test_doc_examples.py
```

Sybil checks `python` fences in public Markdown and doctests in package
source docstrings. Root `README.md` is also an executable source and joins
`tests`, `src/oxyz` and `docs` in default collection. Its example is self-contained
and uses portable Markdown, not native site-only authoring. Plans, skills,
incidental strings in tests and `docs/superpowers/` are not example sources.

Each document gets its own namespace and fresh temporary working directory.
Blocks within a page share names and files in source order; pages cannot depend
on another page. The original working directory is restored even after a failed
example. Ordinary tests do not use this fixture. Existing source docstrings
retain a compatibility `DATA` path to `examples/data`; new Markdown must show
all imports and input setup explicitly, without hidden fixtures or bindings.

## Write a complete example

Use a `python` fence for workflows; add a short title that tells the reader what
the block does. Assert the values that matter rather than merely showing that
a call returns. This example creates its input and checks a read/write round trip
without depending on the checkout directory:

```python title="Verify a small extxyz round trip" linenums="1" hl_lines="7-11 13-17"
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

```python {.no-copy title="Expected Python console output"}
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

## Native presentation

Public docs use Zensical's bundled authoring features. Some directives render
less naturally on GitHub; README remains portable Markdown.

- Add titles with a fence header such as
  `python title="Inspect a trajectory"`. Use `linenums="1"` and `hl_lines="4-6"`
  selectively when they help explain a longer workflow, not on every short block.
- Code copy buttons are enabled globally. Console transcripts and non-executable
  signature notation use a header such as
  `python {.no-copy title="Signature notation"}` to suppress copying.
  `.no-copy` affects the interface, not test execution.
- Use `!!! note`, `!!! tip` and `!!! warning` with an optional quoted title and
  a blank line before four-space-indented prose. Move an important fact into the
  block instead of repeating it outside. Critical limitations remain visible;
  reserve `??? note` collapsible blocks for secondary references.
- Group genuine alternatives with `=== "Package"` and `=== "Source checkout"`,
  indenting each tab's content by four spaces. Keep the core walkthrough linear.
  A tab's visibility never makes Python execution conditional: every collected
  example runs in source order, sharing the page namespace.
- Keep imports and input setup visible. Presentation metadata and indented
  callout/tab fences are checked by the same Sybil Python/doctest evaluators;
  wrong values and failed assertions still report the original document line.

The renderer and Sybil must agree on the chosen fence syntax. After changing
authoring conventions, run the runner regressions as well as affected pages;
successful execution of the remaining blocks is not proof that none were lost.

Markdown lint uses the `mkdocs` flavour for `docs/**/*.md` so it understands
these directives; other Markdown, including README, retains the GFM rules.

## Schematic and optional examples

Runnable examples must not require network access, credentials or large external
corpora. A genuine schematic example may use a named-reason Sybil skip comment:

<!-- skip: next "schematic URL requires private remote credentials and data" -->

```python {.no-copy title="Remote URL syntax — schematic"}
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

## Preview and review

Install the separate, pinned docs group only when rendering the site. From the
checkout root:

```sh title="Build and preview locally"
uv run --group docs zensical build --clean
uv run --group docs zensical serve --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/>. If the port is occupied, choose another loopback
port and record the actual URL; do not stop another process or bind the preview
to the network. `zensical.toml` owns navigation and builds only public pages;
`docs/superpowers/` is excluded from the build. Generated `site/` output and
the renderer cache are not source and must not be committed.

For each content review, run the affected examples and a clean build, then open
the actual pages in a browser. Check navigation, internal links, code/doctest
rendering, tables and narrow-screen overflow. A successful build is not visual
proof. Include the live URL, specific pages and any intentionally schematic
examples when requesting review. Record explicit approval of the reviewed
revision before proceeding past a required review checkpoint; passing checks
or silence are not approval. Changed content needs renewed review.

This is a local preview, not website deployment. README is outside `docs_dir`
and is not rendered by this preview; review it separately as Markdown.
Publication, versioning and generated API documentation are separate work.
See [development](index.md) for setup and [testing](testing.md) for source/wheel
verification boundaries.

## Site styling

The modern theme uses Ledger's ruled layout and Glacier's dark/light palettes,
initially dark. `docs/stylesheets/glacier.css` owns the production cascade,
including self-hosted Atkinson Hyperlegible and Iosevka font faces. Font licences
live beside their files in `docs/fonts/`. `font = false` disables Google Fonts.

Keep code titles, number gutters, highlighted lines and copy controls usable when
changing the stylesheet. Console transcripts and signatures retain `.no-copy`.
Code blocks align with surrounding content and tabs at every screen size.
Check both palettes and narrow screens using actual guides, not just a
typography specimen.

Sidebar groups are collapsible; the current page's group starts open.

## Changelog source

Edit the root `CHANGELOG.md`; `docs/changelog.md` includes it at build time using
Zensical's native `include-markdown` support. Do not maintain a second copy of
release notes. The preview watches the root file for updates. Use absolute
repository URLs for source links so they work both on GitHub and in the site.

## Benchmark evidence

`docs/performance.md` embeds the committed SVGs in `docs/assets/benchmarks/`
and includes `benchmarks/ENVIRONMENT.md`. `docs/reference/benchmarks.md` includes
`benchmarks/RESULTS.md`. Both Markdown files are generated from the same selected
run by `report.py`. The preview watches them; there is no benchmark or
figure-generation step in the docs build.

Update published evidence with the explicit archived input and commands in
[performance work](performance.md#reproduce-the-published-snapshot), not copied
tables or independently edited SVGs. Keep the measured revision and environment
visible, and do not present a historical run as a current-version benchmark.
