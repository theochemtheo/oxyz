from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import yaml

from oxyz import _html, _summary
from oxyz._lookup import NameLookup
from oxyz._schema import (
    KIND_TO_LETTER,
    LETTER_TO_KIND,
    Kind,
    column_sig,
    metadata_sig,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from oxyz._frames import Compression
    from oxyz._remote import StorageOptions

# Whether a schema validates (reports) or projects (reshapes each frame).
Mode = Literal["validate", "project"]


@dataclass(frozen=True, slots=True)
class ColumnRule:
    """One expected per-atom column.

    Attributes
    ----------
    name
        Literal name, a glob (`descriptor_*`), or a regex (`re:...`).
    kind
        Expected value kind.
    width
        Expected number of components per atom (1 for a scalar column).
    required
        For a literal `name`, whether it must be present. For a pattern, the
        default `min` (1 if `required` else 0).
    count
        Exact number of columns a pattern must match, overriding `min`/`max`.
    min
        Lower bound on how many columns a pattern must match.
    max
        Upper bound on how many columns a pattern must match; unbounded if
        unset.
    fill
        Value projection fills an absent column with. REAL defaults to NaN, so
        only an optional INT/BOOL/STR column needs one.
    """

    name: str
    kind: Kind
    width: int = 1
    required: bool = True
    count: int | None = None
    min: int | None = None
    max: int | None = None
    fill: float | int | bool | str | None = None

    def __repr__(self) -> str:
        """Show the fields that differ from their defaults."""
        return _summary.nondefault_repr(self)


@dataclass(frozen=True, slots=True)
class MetadataRule:
    """One expected comment-line entry.

    Attributes
    ----------
    key
        Literal, glob, or regex, as `ColumnRule.name` is.
    kind
        Expected value kind.
    shape
        `()` for a scalar, `(n,)` for an array of length n.
    required
        See `ColumnRule.required`.
    count
        See `ColumnRule.count`.
    min
        See `ColumnRule.min`.
    max
        See `ColumnRule.max`.
    fill
        See `ColumnRule.fill`.
    """

    key: str
    kind: Kind
    shape: tuple[int, ...] = ()
    required: bool = True
    count: int | None = None
    min: int | None = None
    max: int | None = None
    fill: float | int | bool | str | None = None

    def __repr__(self) -> str:
        """Show the fields that differ from their defaults."""
        return _summary.nondefault_repr(self)


@dataclass(frozen=True, slots=True)
class FrameRule:
    """Opt-in structural constraints on a frame; unset fields are not enforced.

    Attributes
    ----------
    n_atoms_min
        Minimum atom count.
    n_atoms_max
        Maximum atom count.
    lattice_required
        Whether a `Lattice` metadata entry must be present.
    """

    n_atoms_min: int | None = None
    n_atoms_max: int | None = None
    lattice_required: bool = False

    def __repr__(self) -> str:
        """Show the fields that differ from their defaults."""
        return _summary.nondefault_repr(self)


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


def _metadata_rule(key: str, attrs: Mapping[str, Any]) -> MetadataRule:
    shape = attrs.get("shape", ())
    return MetadataRule(
        key=key,
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
class SchemaSpec(NameLookup[ColumnRule, MetadataRule]):
    """A prescriptive schema: expected columns, metadata, and structural facts.

    Read and write it with the `from_*`/`to_*` pairs for dicts, JSON, YAML and
    files (`from_file` also reads TOML), or build one directly.
    `spec[name]` returns the rule declared under exactly that name; `rule_for`
    returns the rule that governs a field.

    Attributes
    ----------
    columns
        Expected per-atom columns.
    metadata
        Expected comment-line entries.
    frame
        Structural constraints, or `None`.
    mode
        `"validate"` reports deviations; `"project"` reshapes each frame to the
        declared fields.
    """

    columns: tuple[ColumnRule, ...] = ()
    metadata: tuple[MetadataRule, ...] = ()
    frame: FrameRule | None = None
    mode: Mode = "validate"

    def __repr__(self) -> str:
        """Summarise each rule's type in extxyz letters; `?` marks optional."""
        columns = {
            r.name: column_sig(r.kind, r.width) + ("" if r.required else "?")
            for r in self.columns
        }
        metadata = {
            r.key: metadata_sig(r.kind, r.shape) + ("" if r.required else "?")
            for r in self.metadata
        }
        frame = "" if self.frame is None else f", frame={self.frame!r}"
        mode = "" if self.mode == "validate" else f", mode={self.mode!r}"
        return (
            f"SchemaSpec(columns={_summary.mapping_repr(columns, str)}, "
            f"metadata={_summary.mapping_repr(metadata, str)}{frame}{mode})"
        )

    def _repr_html_(self) -> str:
        """Render as tables for a notebook, previewing a bounded slice of values."""
        return _html.spec(self)

    def _sides(self) -> tuple[dict[str, ColumnRule], dict[str, MetadataRule]]:
        return (
            {r.name: r for r in self.columns},
            {r.key: r for r in self.metadata},
        )

    @overload
    def rule_for(self, name: str, axis: Literal["column"]) -> ColumnRule | None: ...
    @overload
    def rule_for(self, name: str, axis: Literal["metadata"]) -> MetadataRule | None: ...
    @overload
    def rule_for(
        self, name: str, axis: None = None
    ) -> ColumnRule | MetadataRule | None: ...
    def rule_for(
        self, name: str, axis: Literal["column", "metadata"] | None = None
    ) -> ColumnRule | MetadataRule | None:
        """Return the rule validation applies to a field called `name`.

        The literal rule of that name, else the first matching pattern in
        declaration order, else `None`.

        Parameters
        ----------
        name
            A column name or metadata key.
        axis
            `"column"` or `"metadata"` to search one side. `None` searches
            both, and raises `oxyz.AmbiguousNameError` if both govern `name`.

        Returns
        -------
        ColumnRule or MetadataRule or None
            The governing rule, or `None`.
        """
        from oxyz._schema_match import governing_rule

        return governing_rule(self, name, axis)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SchemaSpec:
        """Build a spec from a parsed mapping, as produced by JSON/YAML/TOML.

        Parameters
        ----------
        data
            Mapping with optional `columns`, `metadata`, `frame`, and `mode`
            keys.

        Returns
        -------
        SchemaSpec
            The parsed spec.

        Raises
        ------
        ValueError
            If `mode` is not `"validate"` or `"project"`.
        """
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
    def from_json(cls, text: str) -> SchemaSpec:
        """Parse a spec from JSON text.

        Parameters
        ----------
        text
            JSON document; see `from_dict` for the expected shape.

        Returns
        -------
        SchemaSpec
            The parsed spec.
        """
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_yaml(cls, text: str) -> SchemaSpec:
        """Parse a spec from YAML text.

        Parameters
        ----------
        text
            YAML document; see `from_dict`. An empty document gives a default
            spec.

        Returns
        -------
        SchemaSpec
            The parsed spec.
        """
        return cls.from_dict(yaml.safe_load(text) or {})

    @classmethod
    def from_file(cls, path: str | Path) -> SchemaSpec:
        """Load a spec from a file, dispatching on its extension.

        Parameters
        ----------
        path
            Path to a `.json`, `.yaml`/`.yml`, or `.toml` schema file.

        Returns
        -------
        SchemaSpec
            The parsed spec.

        Raises
        ------
        ValueError
            If `path`'s extension is not one of the supported formats.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        text = path.read_text()
        if suffix == ".json":
            return cls.from_json(text)
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(text)
        if suffix == ".toml":
            return cls.from_dict(tomllib.loads(text))
        raise ValueError(
            f"unsupported schema file extension {suffix!r}; use .json, .yaml, or .toml"
        )

    def to_dict(self) -> dict[str, Any]:
        """Render this spec as a plain mapping, the inverse of `from_dict`.

        Returns
        -------
        dict[str, Any]
            Only non-default fields, so an empty validate-mode spec is `{}`.
        """
        out: dict[str, Any] = {}
        # Only when non-default, so a validate-mode spec renders without it.
        if self.mode != "validate":
            out["mode"] = self.mode
        if self.columns:
            out["columns"] = {rule.name: _column_attrs(rule) for rule in self.columns}
        if self.metadata:
            out["metadata"] = {
                rule.key: _metadata_attrs(rule) for rule in self.metadata
            }
        if self.frame is not None:
            out["frame"] = _frame_attrs(self.frame)
        return out

    def to_json(self) -> str:
        """Render this spec as indented JSON text, the inverse of `from_json`.

        Returns
        -------
        str
            JSON document, 2-space indented.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """Render this spec as YAML text, the inverse of `from_yaml`.

        Returns
        -------
        str
            YAML document with a stable key order.
        """
        return render_yaml(self)

    def to_file(self, path: str | Path) -> None:
        """Write this spec to `path`, dispatching on its extension.

        Parameters
        ----------
        path
            Destination path; must end `.json`, `.yaml`, or `.yml`. There is no
            TOML writer.

        Raises
        ------
        ValueError
            If `path`'s extension is unsupported.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(self.to_json())
        elif suffix in (".yaml", ".yml"):
            path.write_text(self.to_yaml())
        else:
            raise ValueError(
                f"cannot write a schema to a {suffix!r} file; use .json, .yaml, "
                "or .yml (there is no TOML serialiser)"
            )

    def freeze(
        self,
        path: str | Path,
        *,
        compression: Compression = "infer",
        member: str | None = None,
        storage_options: StorageOptions | None = None,
    ) -> SchemaSpec:
        """Expand pattern rules against `path` into a literal, project-ready schema.

        Fields matching in every frame become required, those in only some
        optional. Literal rules pass through unchanged.

        Parameters
        ----------
        path
            File to scan for matching fields.
        compression
            Forces a codec instead of inferring it from `path`; see `oxyz.read`.
        member
            Selects one entry from an archive holding more than one.
        storage_options
            Endpoint/credentials for a remote store; see `oxyz.read`.

        Returns
        -------
        SchemaSpec
            A new spec with `mode="project"`; `self` is untouched.

        Raises
        ------
        SchemaError
            On a matched field whose kind conflicts across frames.
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
    # A 0 or False fill is a real value.
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
    return not name.replace("_", "").isalnum()


def _flow(attrs: Mapping[str, Any]) -> str:
    parts = []
    for key, value in attrs.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        elif isinstance(value, str) and key != "kind":
            # json.dumps gives a double-quoted scalar, so YAML re-reads a string.
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    return "{" + ", ".join(parts) + "}"


def render_yaml(spec: SchemaSpec, notes: Mapping[str, str] | None = None) -> str:
    """Render `spec` as schema YAML, with `notes` as trailing comments by name."""
    notes = notes or {}
    lines: list[str] = []
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
    emit("metadata", {rule.key: _metadata_attrs(rule) for rule in spec.metadata})
    if spec.frame is not None:
        frame_attrs = _frame_attrs(spec.frame)
        if frame_attrs:
            lines.append("frame:")
            for key, value in frame_attrs.items():
                rendered = _flow(value) if isinstance(value, dict) else str(value)
                lines.append(f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"
