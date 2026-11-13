"""Analysis of a union of sibling record classes: merged fields and matching."""

from __future__ import annotations

import functools
import itertools
import types
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAliasType,
    Union,
    final,
    get_args,
    get_origin,
)

from oxyz import _summary
from oxyz._record import FieldSpec, Record

if TYPE_CHECKING:
    from collections.abc import Mapping

    from oxyz._record_fields import FieldType

type Key = tuple[Literal["column", "metadata"], str]
type _Field = tuple[str, FieldType, FieldSpec[Any]]


def members_of(form: object) -> tuple[type[Record], ...]:
    """Return the record classes a `record=` form names, in order, without repeats.

    Raises
    ------
    TypeError
        A member is not a subclass of `Record`.
    """
    found: list[type[Record]] = []
    _collect(form, found)
    return tuple(found)


def _collect(form: object, found: list[type[Record]]) -> None:
    if isinstance(form, TypeAliasType):
        _collect(form.__value__, found)
    elif get_origin(form) in {Union, types.UnionType}:
        for member in get_args(form):
            _collect(member, found)
    elif isinstance(form, type) and issubclass(form, Record) and form is not Record:
        if form not in found:
            found.append(form)
    else:
        name = getattr(form, "__name__", repr(form))
        raise TypeError(f"{name} is not a subclass of Record")


def _key(spec: FieldSpec[Any]) -> Key:
    assert spec.key is not None  # noqa: S101 — type-narrowing only
    return spec.axis, spec.key


def _clause(name: str, required: frozenset[Key], declared: frozenset[Key]) -> str:
    def keys(found: frozenset[Key]) -> str:
        return "{" + ", ".join(sorted(key for _, key in found)) + "}"

    return f"{name}: requires {keys(required)}; may carry {keys(declared - required)}"


@final
@dataclass(frozen=True)
class UnionPlan:
    """The merged fields of a union of records, and how a signature picks a member.

    `matching` holds the keys that tell members apart: every key that is not
    required by all members or filled by all. A member's required matching
    keys are those it declares as neither `X | None` nor with a `fill=`.
    """

    members: tuple[type[Record], ...]
    strict: bool
    fields: tuple[_Field, ...]
    member_fields: tuple[tuple[str, ...], ...]
    names: Mapping[Key, str] = field(compare=False)
    required: frozenset[Key]
    core_filled: frozenset[Key]
    matching: tuple[Key, ...]
    member_required: tuple[frozenset[Key], ...]
    member_declared: tuple[frozenset[Key], ...]
    memo: dict[frozenset[Key], int | None] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def name(self) -> str:
        """The members' class names joined by ` | `."""
        return " | ".join(member.__name__ for member in self.members)

    def member_for(self, signature: frozenset[Key]) -> int | None:
        """Return the index of the one member that admits `signature`, or `None`.

        `signature` is the set of matching keys present in a frame. A member
        admits it when it holds every key the member requires and no key the
        member does not declare.
        """
        if len(self.members) == 1:
            return 0
        if signature not in self.memo:
            self.memo[signature] = next(
                (
                    index
                    for index, (required, declared) in enumerate(
                        zip(self.member_required, self.member_declared, strict=True)
                    )
                    if required <= signature <= declared
                ),
                None,
            )
        return self.memo[signature]

    def describe(self) -> str:
        """List each member with the matching keys it requires and may carry."""
        return "\n".join(
            _clause(member.__name__, required, declared)
            for member, required, declared in zip(
                self.members, self.member_required, self.member_declared, strict=True
            )
        )


@functools.lru_cache(maxsize=128)
def union_plan(members: tuple[type[Record], ...]) -> UnionPlan:
    """Analyse `members`, a tuple of sibling record classes.

    Raises
    ------
    TypeError
        A member subclasses another or differs in `strict`, a key or field
        name is declared inconsistently, or two members admit one signature.
    """
    _check_siblings(members)
    fields, names = _merge(members)
    member_fields = tuple(
        tuple(name for name, _, _ in member.__record_fields__) for member in members
    )
    declared_in = [
        {_key(spec): (ft, spec) for _, ft, spec in m.__record_fields__} for m in members
    ]
    everywhere = set.intersection(*(set(d) for d in declared_in))
    required = frozenset(
        key
        for key in everywhere
        if all(not d[key][0].optional and d[key][1].fill is None for d in declared_in)
    )
    core_filled = frozenset(
        key
        for key in everywhere
        if all(
            not d[key][0].optional and d[key][1].fill is not None for d in declared_in
        )
    )
    matching = tuple(
        key
        for _, _, spec in fields
        if (key := _key(spec)) not in required and key not in core_filled
    )
    in_matching = frozenset(matching)
    member_required = tuple(
        frozenset(
            key
            for key, (ft, spec) in d.items()
            if key in in_matching and not ft.optional and spec.fill is None
        )
        for d in declared_in
    )
    member_declared = tuple(frozenset(d) & in_matching for d in declared_in)
    _check_unambiguous(members, member_required, member_declared)
    return UnionPlan(
        members=members,
        strict=members[0].__record_strict__,
        fields=fields,
        member_fields=member_fields,
        names=names,
        required=required,
        core_filled=core_filled,
        matching=matching,
        member_required=member_required,
        member_declared=member_declared,
    )


def _check_siblings(members: tuple[type[Record], ...]) -> None:
    for a in members:
        for b in members:
            if a is not b and issubclass(a, b):
                raise TypeError(
                    f"{a.__name__} is a subclass of {b.__name__}; "
                    f"union members must be siblings"
                )
    first = members[0]
    for member in members[1:]:
        if member.__record_strict__ != first.__record_strict__:
            raise TypeError(
                f"{first.__name__} and {member.__name__} differ in strict "
                f"({first.__record_strict__} and {member.__record_strict__})"
            )


def _merge(
    members: tuple[type[Record], ...],
) -> tuple[tuple[_Field, ...], dict[Key, str]]:
    merged: dict[Key, tuple[_Field, type[Record]]] = {}
    by_name: dict[str, tuple[Key, type[Record]]] = {}
    for member in members:
        for entry in member.__record_fields__:
            name, ft, spec = entry
            key = _key(spec)
            owner = by_name.setdefault(name, (key, member))
            if owner[0] != key:
                raise TypeError(
                    f"{owner[1].__name__} and {member.__name__} read field "
                    f"{name!r} from different keys: {owner[0][1]!r} and {key[1]!r}"
                )
            if key not in merged:
                merged[key] = (entry, member)
                continue
            (first_name, first_ft, first_spec), first = merged[key]
            if first_name != name:
                raise TypeError(
                    f"{first.__name__}.{first_name} and {member.__name__}.{name} "
                    f"both read {key[0]} key {key[1]!r}"
                )
            if first_ft != ft or not _summary.values_equal(first_spec.fill, spec.fill):
                raise TypeError(
                    f"{first.__name__} and {member.__name__} declare {key[0]} key "
                    f"{key[1]!r} (field {name!r}) differently: "
                    f"{_describe(first_ft, first_spec)} and {_describe(ft, spec)}"
                )
    fields = tuple(entry for entry, _ in merged.values())
    names = {key: entry[0] for key, (entry, _) in merged.items()}
    return fields, names


def _describe(ft: FieldType, spec: FieldSpec[Any]) -> str:
    optional = ", optional" if ft.optional else ""
    return f"{ft.signature()}{optional}, fill={spec.fill!r}"


def _check_unambiguous(
    members: tuple[type[Record], ...],
    required: tuple[frozenset[Key], ...],
    declared: tuple[frozenset[Key], ...],
) -> None:
    for i, j in itertools.combinations(range(len(members)), 2):
        if required[i] <= declared[j] and required[j] <= declared[i]:
            a, b = members[i].__name__, members[j].__name__
            signature = ", ".join(sorted(key for _, key in required[i] | required[j]))
            raise TypeError(
                f"{a} and {b} both admit a frame whose matching keys are "
                f"{{{signature}}}: {_clause(a, required[i], declared[i])}; "
                f"{_clause(b, required[j], declared[j])}"
            )
