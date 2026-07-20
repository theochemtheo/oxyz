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
        For a literal `name`, whether it must be present. For a pattern
        rule, the default lower bound (1 if `required` else 0) used when
        `min` is unset.
    count
        Exact number of columns a pattern must match, overriding `min`/`max`.
        Ignored for a literal name (which matches 0 or 1).
    min
        Lower bound on how many columns a pattern must match. Ignored for a
        literal name.
    max
        Upper bound on how many columns a pattern must match, or unbounded if
        unset. Ignored for a literal name.
    fill
        Value projection fills this column with when it is absent. Only used
        under project mode; REAL defaults to NaN there, so a fill is required
        only for an optional INT/BOOL/STR column (which has no natural null).
    """

    name: str
    kind: Kind
    width: int = 1
    required: bool = True
    count: int | None = None
    min: int | None = None
    max: int | None = None
    fill: float | int | bool | str | None = None


@dataclass(frozen=True, slots=True)
class MetadataRule:
    """One expected comment-line entry.

    Attributes
    ----------
    key
        Literal, glob, or regex, as `ColumnRule.name` is (the metadata
        identifier is `key`, matching `MetadataSchema.key`).
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


@dataclass(frozen=True, slots=True)
class FrameRule:
    """Opt-in structural constraints on a frame. Any field left unset is not enforced.

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
class SchemaSpec:
    """A prescriptive schema: expected columns, metadata, and structural facts.

    Each format has a `from_`/`to_` pair: `from_dict`/`to_dict`,
    `from_json`/`to_json`, `from_yaml`/`to_yaml` (`to_yaml` is hand-templated for
    stable order and comment support), and `from_file`/`to_file` (dispatching on
    `.json`/`.yaml`/`.yml`, with `from_file` also reading `.toml`). Or build one
    directly.

    Attributes
    ----------
    columns
        Expected per-atom columns.
    metadata
        Expected comment-line entries.
    frame
        Structural constraints, or `None` if none apply.
    mode
        Whether the spec validates (report only) or projects (reshape each
        frame to the declared field set).
    """

    columns: tuple[ColumnRule, ...] = ()
    metadata: tuple[MetadataRule, ...] = ()
    frame: FrameRule | None = None
    mode: Mode = "validate"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SchemaSpec:
        """Build a spec from a parsed mapping, as produced by JSON/YAML/TOML.

        Parameters
        ----------
        data
            Mapping with optional `columns`, `metadata`, `frame`, and `mode`
            keys, in the schema file format.

        Returns
        -------
        SchemaSpec
            The parsed spec.

        Raises
        ------
        ValueError
            If `mode` is set to anything other than `"validate"` or
            `"project"`.
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
            YAML document; see `from_dict` for the expected shape. An empty
            document parses to an all-default spec.

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
            Only non-default fields are present, so a `mode="validate"` spec
            with no columns/metadata/frame renders as `{}`.
        """
        out: dict[str, Any] = {}
        # Emitted first, and only when non-default, so validate-mode specs
        # round-trip byte-identically to before projection existed.
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
            YAML document with a hand-templated, stable key order.
        """
        return render_yaml(self)

    def to_file(self, path: str | Path) -> None:
        """Write this spec to `path`, dispatching on its extension.

        TOML output is rejected — there is no TOML serialiser, and writing
        YAML under a `.toml` name would produce a file that will not re-read.

        Parameters
        ----------
        path
            Destination path; must end `.json`, `.yaml`, or `.yml`.

        Raises
        ------
        ValueError
            If `path`'s extension is `.toml` or otherwise unsupported.
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

        The result has `mode='project'`. Columns matching in every frame
        become required; those in only some become optional, so projection
        fills them. Literal rules pass through unchanged.

        Parameters
        ----------
        path
            File to scan for matching fields.
        compression
            Forces a codec instead of inferring it from `path`.
        member
            Selects one entry from a `.zip`/`.tar`/`.tar.gz` holding more
            than one.
        storage_options
            Endpoint/credentials for a remote store, falling back to `AWS_*`
            env vars.

        Returns
        -------
        SchemaSpec
            A new spec with patterns expanded to literals; `self` is
            untouched.

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
            # re-reads it as a string, not as a bool / number / null. json.dumps
            # yields a valid double-quoted scalar, escaping any `"` or `\`. The
            # kind letter is a controlled R/I/L/S token and stays bare.
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        parts.append(f"{key}: {rendered}")
    return "{" + ", ".join(parts) + "}"


def render_yaml(spec: SchemaSpec, notes: Mapping[str, str] | None = None) -> str:
    """Render `spec` as schema YAML.

    `notes` maps an entry name to a trailing `# comment` (used by emission to
    flag drift); comments are dropped on reload, so the text always parses
    back to `spec`.
    """
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
    emit("metadata", {rule.key: _metadata_attrs(rule) for rule in spec.metadata})
    if spec.frame is not None:
        frame_attrs = _frame_attrs(spec.frame)
        if frame_attrs:
            lines.append("frame:")
            for key, value in frame_attrs.items():
                rendered = _flow(value) if isinstance(value, dict) else str(value)
                lines.append(f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"
