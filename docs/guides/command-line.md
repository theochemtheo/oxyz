# Command line

Installing oxyz provides the `oxyz` command. `python -m oxyz` uses the same
entry point; `uvx oxyz` runs a released package without a persistent installation.
From a source checkout use `uv run oxyz` so the compiled extension stays current.

## Inspect the available commands

```sh title="Inspect the command-line interface"
oxyz --help
oxyz scan --help
oxyz check --help
oxyz freeze --help
```

| Command | Work performed | Output |
| --- | --- | --- |
| `scan PATH` | Parse the whole file and infer its schema | Atom-count statistics and pasteable schema syntax |
| `scan PATH --no-schema` | Scan frame boundaries and declared counts, without parsing atom values | Structural statistics only |
| `check PATH --schema SPEC` | Validate the complete input against the spec | Grouped deviations, or a conformant summary |
| `freeze PATH --schema SPEC --out OUT` | Expand spec patterns against a representative sample | A literal, project-mode schema file |

!!! note "The command and Python scan do different work"

    `oxyz scan` infers a schema by default. Python's `oxyz.scan()` is the cheap
    structural operation. `--no-schema` is not evidence that atom values parse
    or satisfy a schema. See [storage and streaming](../concepts/storage-and-streaming.md).

## Inspect, declare, check and freeze

These POSIX-shell commands create their own small input. Run them in a scratch
working directory; the generated schemas are output files, not bundled fixtures.

```sh title="Create a local two-frame trajectory"
cat > sample.extxyz <<'EOF'
2
Properties=species:S:1:pos:R:3 energy=-1.5
H 0.0 0.0 0.0
O 1.0 0.0 0.0
2
Properties=species:S:1:pos:R:3 energy=-1.7
H 0.0 0.0 0.0
O 1.1 0.0 0.0
EOF
```

```sh title="Inspect and save a schema"
oxyz scan sample.extxyz
oxyz scan sample.extxyz --no-schema --json
oxyz scan sample.extxyz --emit-schema schema.yaml
oxyz check sample.extxyz --schema schema.yaml --conformance strict
oxyz freeze sample.extxyz --schema schema.yaml --out frozen.yaml
oxyz scan sample.extxyz --emit-schema projected.json --project
```

The input has two frames, four atoms in total and two atoms per frame.
`schema.yaml` describes `species:S:1`, `pos:R:3` and scalar REAL `energy`.
The strict check exits zero. `frozen.yaml` and `projected.json` have
`mode: project` and literal field names; neither command writes a transformed
trajectory. Use a Python reader with the saved spec to project values.

`--emit-schema` writes the spec instead of the scan summary. It supports JSON
or YAML; `freeze --out` supports `.json`, `.yaml` and `.yml`, not TOML output.
Input specs may be JSON, YAML or TOML. `--project` requires `--emit-schema`,
and `--emit-schema` cannot be combined with `--no-schema`.

Review inferred and frozen rules before adopting them. Freeze expands observed
names; it does not prove the pattern's signature or cardinality, or a contract
for unseen data. See [schemas and projection](schemas-and-projection.md) for
requiredness, fills, omission and promotion.

## Machine-readable reports and exit status

`scan --json` emits an object with `stats` and, unless `--no-schema` was given,
`schema`. `stats` contains `n_frames`, `total_atoms`, `min_atoms`, `max_atoms`,
`mean_atoms`, `median_atoms` and `std_atoms`. The observed schema report is not
itself the loadable schema format: use `--emit-schema` for a `SchemaSpec` file.

```sh title="Obtain a structured validation report"
oxyz check sample.extxyz --schema schema.yaml --json
```

`check --json` includes `path`, `conformance`, `n_frames` and `violations`.
Identical deviations are grouped by axis, name, deviation and expected/found
signature. Each group includes `frames`, `first_frame` and `first_line`.
`first_frame` is zero-based; `first_line` is the one-based start of the first
affected frame, not the offending token's line or column. A parse error stops
the traversal rather than producing a complete schema-deviation report.

| Exit status | Meaning |
| --- | --- |
| `0` | Successful scan/freeze, or a conformant check |
| `1` | A check found deviations, or a caught input/schema/I/O error |
| `2` | Missing command or argparse usage error, such as an unknown flag |

!!! warning "CLI warn still fails a check with deviations"

    For `check`, `required` permits extras; `strict` and `warn` report them.
    All three report missing required fields, mismatches and pattern counts.
    `--conformance warn` still exits **1** when deviations exist. It does not
    turn failed validation into success. `check` validates without projecting
    or rewriting the file, even when the loaded spec says `mode: project`.

An operational error is printed to stderr with an `oxyz:` prefix. Do not assume
`--json` makes every error a JSON object. Python's structured error attributes
are described in the [error reference](../reference/errors.md).

## Compressed and remote inputs

All three commands accept `--compression` with `infer`, `none`, `gzip`, `zstd`,
`zip`, `tar`, `tar.gz` or `tar.zst`. Select an archive entry with `--member NAME`,
or join all extxyz entries in archive order with `--all-members`; the two flags
are mutually exclusive.

```sh title="Inspect all local archive members"
oxyz scan runs.zip --all-members --no-schema --json
```

Here `runs.zip` is an existing archive, not created by the two-frame example.
The [compressed and remote guide](compressed-and-remote.md) creates local
archives and explains logical-stream error context and seek restrictions.

For remote sources install `oxyz[s3]`; all three commands accept repeatable
`--storage-option KEY=VALUE`. Values supplied by this flag are strings, not a
way to pass arbitrary nested Python dictionaries. Prefer the documented
environment variables for credentials rather than exposing secrets in shell
history or process arguments. S3-compatible endpoints, HuggingFace URL/token
syntax and best-effort providers are covered in that same guide.

## Verification and ownership

Shell fences are instructions, not Sybil-executed Python examples. The
[CLI tests](https://github.com/theochemtheo/oxyz/blob/main/tests/test_cli.py)
cover scan/check/freeze outputs and failures; the documentation workflow also
exercises the installed command in a temporary directory, checking emitted
schemas, JSON values and exit codes. This does not claim live remote coverage.

[`_cli.py`](https://github.com/theochemtheo/oxyz/blob/main/src/oxyz/_cli.py)
owns arguments and reports. The underlying schema and source contracts are
shared with the library, not reimplemented for the shell.
