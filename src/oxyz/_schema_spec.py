from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from oxyz._schema import Kind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from oxyz._frames import Compression
    from oxyz._remote import StorageOptions

# Whether a schema validates (report only) or projects (reshape each frame to
# the declared field set). See oxyz._project for what project mode compiles to.
Mode = Literal["validate", "project"]

LETTER_TO_KIND: dict[str, Kind] = {
    "R": Kind.REAL,
    "I": Kind.INT,
    "L": Kind.BOOL,
    "S": Kind.STR,
}
KIND_TO_LETTER: dict[Kind, str] = {
    kind: letter for letter, kind in LETTER_TO_KIND.items()
}


@dataclass(frozen=True, slots=True)
class ColumnRule:
    """One expected per-atom column. `name` is a literal, a glob (`descriptor_*`),
    or a regex (`re:...`). `count`/`min`/`max` bound how many columns a pattern
    matches; they are ignored for a literal name (which matches 0 or 1)."""

    name: str
    kind: Kind
    width: int = 1
    required: bool = True
    count: int | None = None
    min: int | None = None
    max: int | None = None
    # The value projection fills this column with when it is absent. Only used
    # under project mode; REAL defaults to NaN there, so a fill is required only
    # for an optional INT/BOOL/STR column (which has no natural null).
    fill: float | int | bool | str | None = None


@dataclass(frozen=True, slots=True)
class MetadataRule:
    """One expected comment-line key. `shape` is `()` for a scalar, `(n,)` for an
    array of length n. Pattern cardinality works as for `ColumnRule`."""

    name: str
    kind: Kind
    shape: tuple[int, ...] = ()
    required: bool = True
    count: int | None = None
    min: int | None = None
    max: int | None = None
    # See ColumnRule.fill.
    fill: float | int | bool | str | None = None


@dataclass(frozen=True, slots=True)
class FrameRule:
    """Opt-in structural constraints. Any field left unset is not enforced."""

    n_atoms_min: int | None = None
    n_atoms_max: int | None = None
    lattice_required: bool = False


def _kind(letter: object) -> Kind:
    try:
        return LETTER_TO_KIND[str(letter)]
    except KeyError:
        raise ValueError(f"unknown kind {letter!r}; use one of R, I, L, S") from None


def _column_rule(name: str, attrs: Mapping[str, Any]) -> ColumnRule:
    return ColumnRule(
        name=name,
        kind=_kind(attrs["kind"]),
        width=int(attrs.get("width", 1)),
        required=bool(attrs.get("required", True)),
        count=attrs.get("count"),
        min=attrs.get("min"),
        max=attrs.get("max"),
        fill=attrs.get("fill"),
    )


def _metadata_rule(name: str, attrs: Mapping[str, Any]) -> MetadataRule:
    shape = attrs.get("shape", ())
    return MetadataRule(
        name=name,
        kind=_kind(attrs["kind"]),
        shape=tuple(int(n) for n in shape),
        required=bool(attrs.get("required", True)),
        count=attrs.get("count"),
        min=attrs.get("min"),
        max=attrs.get("max"),
        fill=attrs.get("fill"),
    )


def _frame_rule(attrs: Mapping[str, Any]) -> FrameRule:
    n_atoms = attrs.get("n_atoms", {})
    lattice = attrs.get("lattice")
    return FrameRule(
        n_atoms_min=n_atoms.get("min"),
        n_atoms_max=n_atoms.get("max"),
        lattice_required=lattice in (True, "required"),
    )


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    """A prescriptive schema: expected columns, metadata, and structural facts.

    Build one from a mapping (`from_dict`), a file (`from_file`, dispatching on
    `.json`/`.yaml`/`.yml`/`.toml`), or directly. Serialise with `to_yaml`
    (hand-templated for stable order and comment support) or `to_json`.
    """

    columns: tuple[ColumnRule, ...] = ()
    metadata: tuple[MetadataRule, ...] = ()
    frame: FrameRule | None = None
    mode: Mode = "validate"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SchemaSpec:
        columns = tuple(
            _column_rule(name, attrs) for name, attrs in data.get("columns", {}).items()
        )
        metadata = tuple(
            _metadata_rule(name, attrs)
            for name, attrs in data.get("metadata", {}).items()
        )
        frame = _frame_rule(data["frame"]) if "frame" in data else None
        mode = data.get("mode", "validate")
        if mode not in ("validate", "project"):
            raise ValueError(
                f"unknown schema mode {mode!r}; use 'validate' or 'project'"
            )
        return cls(columns=columns, metadata=metadata, frame=frame, mode=mode)

    @classmethod
    def from_yaml_text(cls, text: str) -> SchemaSpec:
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def from_file(cls, path: str | Path) -> SchemaSpec:
        path = Path(path)
        suffix = path.suffix.lower()
        text = path.read_text()
        if suffix == ".json":
            return cls.from_dict(json.loads(text))
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml_text(text)
        if suffix == ".toml":
            return cls.from_dict(tomllib.loads(text))
        raise ValueError(
            f"unsupported schema file extension {suffix!r}; use .json, .yaml, or .toml"
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        # Emitted first, and only when non-default, so validate-mode specs
        # round-trip byte-identically to before projection existed.
        if self.mode != "validate":
            out["mode"] = self.mode
        if self.columns:
            out["columns"] = {rule.name: _column_attrs(rule) for rule in self.columns}
        if self.metadata:
            out["metadata"] = {
                rule.name: _metadata_attrs(rule) for rule in self.metadata
            }
        if self.frame is not None:
            out["frame"] = _frame_attrs(self.frame)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        return render_yaml(self)

    def freeze(
        self,
        path: str | Path,
        *,
        compression: Compression = "infer",
        member: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> SchemaSpec:
        """Expand this schema's pattern rules against `path` into a literal,
        project-ready schema (`mode='project'`).

        Columns matching in every frame become required; those in only some
        become optional, so projection fills them. Raises `SchemaError` on a
        matched field whose kind conflicts across frames. Literal rules pass
        through unchanged. Returns a new spec; `self` is untouched.
        """
        from oxyz._project import freeze_spec

        return freeze_spec(
            self,
            path,
            compression=compression,
            member=member,
            storage_options=storage_options,
        )


def _column_attrs(rule: ColumnRule) -> dict[str, Any]:
    attrs: dict[str, Any] = {"kind": KIND_TO_LETTER[rule.kind]}
    if rule.width != 1:
        attrs["width"] = rule.width
    if not rule.required:
        attrs["required"] = False
    for key in ("count", "min", "max"):
        value = getattr(rule, key)
        if value is not None:
            attrs[key] = value
    # `is not None`, not truthiness: a 0 / False fill is a legitimate value.
    if rule.fill is not None:
        attrs["fill"] = rule.fill
    return attrs


def _metadata_attrs(rule: MetadataRule) -> dict[str, Any]:
    attrs: dict[str, Any] = {"kind": KIND_TO_LETTER[rule.kind]}
    if rule.shape:
        attrs["shape"] = list(rule.shape)
    if not rule.required:
        attrs["required"] = False
    for key in ("count", "min", "max"):
        value = getattr(rule, key)
        if value is not None:
            attrs[key] = value
    if rule.fill is not None:
        attrs["fill"] = rule.fill
    return attrs


def _frame_attrs(frame: FrameRule) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    n_atoms: dict[str, int] = {}
    if frame.n_atoms_min is not None:
        n_atoms["min"] = frame.n_atoms_min
    if frame.n_atoms_max is not None:
        n_atoms["max"] = frame.n_atoms_max
    if n_atoms:
        attrs["n_atoms"] = n_atoms
    if frame.lattice_required:
        attrs["lattice"] = "required"
    return attrs


def _needs_quotes(name: str) -> bool:
    # Quote anything that is not a plain YAML identifier (globs, `re:` regexes).
    return not name.replace("_", "").isalnum()


def _flow(attrs: Mapping[str, Any]) -> str:
    parts = []
    for key, value in attrs.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        elif isinstance(value, str) and key != "kind":
            # Quote an arbitrary string value (e.g. a string `fill`) so YAML
            # re-reads it as a string, not as a bool / number / null. The kind
            # letter is a controlled R/I/L/S token and stays bare.
            rendered = f'"{value}"'
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    return "{" + ", ".join(parts) + "}"


def render_yaml(spec: SchemaSpec, notes: Mapping[str, str] | None = None) -> str:
    """Render `spec` as schema YAML. `notes` maps an entry name to a trailing
    `# comment` (used by emission to flag drift); comments are dropped on reload,
    so the text always parses back to `spec`."""

    notes = notes or {}
    lines: list[str] = []
    # Non-default mode renders first, matching to_dict's key order.
    if spec.mode != "validate":
        lines.append(f"mode: {spec.mode}")

    def emit(section: str, entries: dict[str, dict[str, Any]]) -> None:
        if not entries:
            return
        lines.append(f"{section}:")
        for name, attrs in entries.items():
            key = f'"{name}"' if _needs_quotes(name) else name
            line = f"  {key}: {_flow(attrs)}"
            if name in notes:
                line += f"  # {notes[name]}"
            lines.append(line)

    emit("columns", {rule.name: _column_attrs(rule) for rule in spec.columns})
    emit("metadata", {rule.name: _metadata_attrs(rule) for rule in spec.metadata})
    if spec.frame is not None:
        frame_attrs = _frame_attrs(spec.frame)
        if frame_attrs:
            lines.append("frame:")
            for key, value in frame_attrs.items():
                rendered = _flow(value) if isinstance(value, dict) else str(value)
                lines.append(f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"
