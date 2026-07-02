from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from oxyz import infer_schema, scan

if TYPE_CHECKING:
    from oxyz._scan import FrameIndex
    from oxyz._schema import Schema

    # scan and infer_schema both report atom-count statistics; the schema adds
    # the column/metadata detail.
    StatsSource = FrameIndex | Schema


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``oxyz`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return func(args)
    except (ValueError, OSError) as exc:
        # ValueError covers ParseError and the archive/member selection errors;
        # OSError covers missing/unreadable files.
        print(f"oxyz: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oxyz", description="Inspect extxyz/xyz files."
    )
    subparsers = parser.add_subparsers(metavar="<command>")
    _add_scan_parser(subparsers)
    return parser


def _add_scan_parser(subparsers: argparse._SubParsersAction) -> None:
    # The CLI `scan` is a human-facing summary that parses the whole file
    # (distribution stats from oxyz.scan, plus the inferred schema) -- distinct
    # from the oxyz.scan() primitive, which parses nothing. --no-schema drops
    # back to that cheap path. Running both passes is deliberate; folding the
    # distribution stats into infer_schema (one pass) is a future core change.
    scan_parser = subparsers.add_parser(
        "scan",
        help="summarise a file's frames and inferred schema",
        description=(
            "Summarise a file: per-frame atom-count statistics and, unless "
            "--no-schema is given, the inferred schema. Reads the whole file."
        ),
    )
    scan_parser.add_argument(
        "path", help="path to an extxyz/xyz file (compressed forms are read too)"
    )
    scan_parser.add_argument(
        "--no-schema",
        action="store_true",
        help="skip schema inference; report only the cheap structural scan",
    )
    scan_parser.add_argument(
        "--json", action="store_true", help="emit a JSON object instead of text"
    )
    scan_parser.add_argument(
        "--compression",
        choices=("infer", "none", "gzip", "zstd", "zip"),
        default="infer",
        help="codec to read PATH as (default: infer from the name)",
    )
    scan_parser.add_argument(
        "--member",
        default=None,
        help="entry to read from a multi-member archive (.zip/.tar/.tar.gz)",
    )
    scan_parser.add_argument(
        "--storage-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="storage_options",
        help="remote store option (repeatable), e.g. endpoint=..., region=...",
    )
    scan_parser.add_argument(
        "--emit-schema",
        default=None,
        metavar="PATH",
        dest="emit_schema",
        help=(
            "write the inferred schema to PATH (.yaml or .json) instead of "
            "the text summary"
        ),
    )
    scan_parser.set_defaults(func=_cmd_scan)


def _parse_storage_options(items: list[str]) -> dict[str, str] | None:
    if not items:
        return None
    options: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--storage-option must be KEY=VALUE, got {item!r}")
        options[key] = value
    return options


def _cmd_scan(args: argparse.Namespace) -> int:
    # The schema pass keeps the per-frame atom counts, so it yields the same
    # distribution stats as scan -- run only one. --no-schema wants no parse
    # at all, so it falls back to the cheap structural scan.
    storage_options = _parse_storage_options(args.storage_options)
    if args.no_schema:
        stats: StatsSource = scan(
            args.path,
            compression=args.compression,
            member=args.member,
            storage_options=storage_options,
        )
        schema = None
    else:
        schema = infer_schema(
            args.path,
            compression=args.compression,
            member=args.member,
            storage_options=storage_options,
        )
        stats = schema
    if args.emit_schema is not None:
        if schema is None:
            raise ValueError("--emit-schema needs the schema pass; drop --no-schema")
        _write_schema(schema, Path(args.emit_schema))
        return 0
    if args.json:
        print(json.dumps(_scan_payload(stats, schema), indent=2))
    else:
        _print_scan_summary(stats, schema)
    return 0


def _scan_payload(stats: StatsSource, schema: Schema | None) -> dict:
    payload: dict = {"stats": _stats_dict(stats)}
    if schema is not None:
        # asdict serialises the nested schema dataclasses; drop the cached
        # report (its text form lives in the non-JSON path) and the raw
        # per-frame counts (the derived stats already stand for them).
        schema_dict = dataclasses.asdict(schema)
        schema_dict.pop("_report", None)
        schema_dict.pop("n_atoms", None)
        payload["schema"] = schema_dict
    return payload


def _stats_dict(stats: StatsSource) -> dict:
    return {
        "n_frames": stats.n_frames,
        "total_atoms": stats.total_atoms,
        "min_atoms": stats.min_atoms,
        "max_atoms": stats.max_atoms,
        "mean_atoms": stats.mean_atoms,
        "median_atoms": stats.median_atoms,
        "std_atoms": stats.std_atoms,
    }


def _write_schema(schema: Schema, path: Path) -> None:
    spec = schema.to_spec()
    if path.suffix.lower() == ".json":
        path.write_text(spec.to_json())
    else:
        from oxyz._schema_emit import spec_and_notes
        from oxyz._schema_spec import render_yaml

        spec, notes = spec_and_notes(schema)
        path.write_text(render_yaml(spec, notes))


def _schema_block(schema: Schema) -> str:
    from oxyz._schema_emit import spec_and_notes
    from oxyz._schema_spec import render_yaml

    spec, notes = spec_and_notes(schema)
    return render_yaml(spec, notes)


def _print_scan_summary(stats: StatsSource, schema: Schema | None) -> None:
    print(f"frames:      {stats.n_frames}")
    if stats.n_frames:
        print(f"atoms total: {stats.total_atoms}")
        print(
            f"atoms/frame: min {stats.min_atoms}  max {stats.max_atoms}  "
            f"mean {stats.mean_atoms:.2f}  median {stats.median_atoms:.2f}  "
            f"std {stats.std_atoms:.2f}"
        )
    if schema is not None:
        print()
        print(
            "# schema — paste into a .yaml and read with read_frames(..., schema=...)"
        )
        print(_schema_block(schema).rstrip("\n"))
