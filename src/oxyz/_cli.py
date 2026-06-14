from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import TYPE_CHECKING

from oxyz import ParseError, infer_schema, scan

if TYPE_CHECKING:
    from oxyz._scan import FrameIndex
    from oxyz._schema import Schema


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
    except (ParseError, OSError) as exc:
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
    scan_parser.add_argument("path", help="path to an extxyz/xyz file")
    scan_parser.add_argument(
        "--no-schema",
        action="store_true",
        help="skip schema inference; report only the cheap structural scan",
    )
    scan_parser.add_argument(
        "--json", action="store_true", help="emit a JSON object instead of text"
    )
    scan_parser.set_defaults(func=_cmd_scan)


def _cmd_scan(args: argparse.Namespace) -> int:
    index = scan(args.path)
    schema = None if args.no_schema else infer_schema(args.path)
    if args.json:
        print(json.dumps(_scan_payload(index, schema), indent=2))
    else:
        _print_scan_summary(index, schema)
    return 0


def _scan_payload(index: FrameIndex, schema: Schema | None) -> dict:
    payload: dict = {"stats": _stats_dict(index)}
    if schema is not None:
        # asdict serialises the nested schema dataclasses; drop the cached
        # human-readable report -- the text form lives in the non-JSON path.
        schema_dict = dataclasses.asdict(schema)
        schema_dict.pop("_report", None)
        payload["schema"] = schema_dict
    return payload


def _stats_dict(index: FrameIndex) -> dict:
    return {
        "n_frames": index.n_frames,
        "total_atoms": index.total_atoms,
        "min_atoms": index.min_atoms,
        "max_atoms": index.max_atoms,
        "mean_atoms": index.mean_atoms,
        "median_atoms": index.median_atoms,
        "std_atoms": index.std_atoms,
    }


def _print_scan_summary(index: FrameIndex, schema: Schema | None) -> None:
    print(f"frames:      {index.n_frames}")
    if index.n_frames:
        print(f"atoms total: {index.total_atoms}")
        print(
            f"atoms/frame: min {index.min_atoms}  max {index.max_atoms}  "
            f"mean {index.mean_atoms:.2f}  median {index.median_atoms:.2f}  "
            f"std {index.std_atoms:.2f}"
        )
    if schema is not None:
        print()
        print(schema.report().rstrip("\n"))
