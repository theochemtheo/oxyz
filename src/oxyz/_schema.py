from __future__ import annotations

from pathlib import Path

import oxyz._rust as _rust


def infer_schema(path: str | Path) -> str:
    """Infer the dataset's schema and return a human-readable report.

    Provisional API: returns text only while the schema shape settles;
    structured access from Python will replace this.
    """
    return _rust.infer_schema(str(path))
