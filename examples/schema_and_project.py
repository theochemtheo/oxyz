"""Validate a file against a schema, then project a mixed-schema file to batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import oxyz

MIXED = Path(__file__).parent / "data" / "mixed.extxyz"


def main() -> None:
    # mixed.extxyz has `forces` in only some frames, so it does not batch as-is.
    schema = oxyz.infer_schema(MIXED)
    print("consistent:", schema.is_consistent)  # False

    # Declare a fixed shape and project: forces optional, absent -> NaN-filled.
    spec = oxyz.SchemaSpec.from_dict(
        {
            "columns": {
                "species": {"kind": "S"},
                "pos": {"kind": "R", "width": 3},
                "forces": {"kind": "R", "width": 3, "required": False},
            },
            "mode": "project",
        }
    )
    batch = oxyz.read_batch(MIXED, schema=spec)  # now batchable
    print("batched atoms:", np.asarray(batch.columns["pos"]).shape[0])
    print("forces column shape:", np.asarray(batch.columns["forces"]).shape)


if __name__ == "__main__":
    main()
