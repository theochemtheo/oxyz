# Extended XYZ format

oxyz reads extended XYZ into typed columns and frame metadata. This page describes
its supported syntax, not every dialect called XYZ. See [compatibility](compatibility.md)
for ASE interpretation, [the data model](../concepts/data-model.md) for Python
representations and [errors](errors.md) for diagnostics.

## Frame structure

Each frame consists of a non-negative integer atom count, one comment/header line,
and exactly that many atom lines. Counts are trimmed before parsing. LF and CRLF
line endings are supported; a final atom line need not end with a newline.
Zero-atom frames still require a header.

A blank count line terminates reading rather than separating frames. Consequently,
trailing blank lines are tolerated, a leading blank line yields no frames, and a
blank line between frames hides the remaining frames. A non-blank invalid count
or a truncated frame is an error, not an end marker.

!!! warning "Properties is required, even for .xyz"

    The header consists of whitespace-separated `key=value` pairs. Do not put
    spaces around `=`. Arbitrary free-text comments and standalone flags are not
    supported. `Properties` is required: there is no implicit
    `species:S:1:pos:R:3` fallback for ordinary XYZ. The `.xyz` extension does not
    change this requirement.

## Properties and atom values

`Properties=species:S:1:pos:R:3:id:I:1` declares consecutive
`name:kind:width` triplets. Names must be non-empty; kind letters are case-sensitive.
Widths must be integers from 1 through 65536. Each atom line must contain exactly
the sum of the declared widths in ASCII-whitespace-separated tokens.

| Kind | Atom value | Python column |
| --- | --- | --- |
| `R` | Real number, using decimal/exponent syntax | `float64` NumPy array |
| `I` | Signed 64-bit integer | `int64` NumPy array |
| `L` | `T`, `TRUE`, `True`, `true`, `1`, or the corresponding `F`, `FALSE`, `False`, `false`, `0` | Boolean NumPy array |
| `S` | UTF-8 token | List of strings, or nested lists for width greater than one |

Width one gives a one-dimensional column; larger widths give shape
`(n_atoms, width)`. Atom strings are tokens, not comment-line quoted values:
quotes do not group whitespace on atom lines. Lower-case single-letter `t`/`f`
are accepted in metadata, but not in `L` atom cells. Fortran `d`/`D` exponents
are a metadata scalar feature, not an atom-real syntax guarantee.

The first `Properties` entry defines columns and is consumed rather than copied
to metadata. Column names are not renamed or chemically validated: `species`,
`Z`, `REF_forces` and arbitrary user names are ordinary native columns.

## Metadata inference

Metadata has no explicit kind declaration. Values are inferred in this order:
integer, real, boolean, then string. Integers have optional signs and decimal
leading zeros (`012` means 12). Real scalars have a decimal point or exponent;
`e`, `E`, `d` and `D` exponents are recognised. Thus `1` is an integer and
`1.0` is real. A syntactically valid token that is not numeric, such as `1.3k7`,
remains a string rather than raising a numeric conversion error.

Metadata booleans are exactly `t`, `T`, `true`, `True`, `TRUE` and
`f`, `F`, `false`, `False`, `FALSE`. `TRuE` is a string. `0` and `1` are
integers, not booleans.

### Quoting

Only double quotes group a value containing whitespace. Outer double quotes are
removed, but they do not force a string: `step=" 3 "` is integer 3.
String interiors retain their padding: `label=" padded "` retains both spaces.
Single quotes are ordinary characters: `label='hello'` retains the apostrophes,
and `note=it's` retains its apostrophe. They cannot group a multi-word value.

A bare value cannot contain `=`, `"`, `,`, `[`, `]`, `{`, `}` or a backslash.
Double-quoted strings can contain reserved characters such as `=` and commas.
Do not rely on backslash escaping to embed double quotes: the header splitter
ends a quoted span at the next double quote.

### Arrays and brace groups

| Syntax | Interpretation |
| --- | --- |
| `v="1 2 3"` or `v={1 2 3}` | One-dimensional integer array |
| `v="1 2.5"` | One-dimensional real array |
| `v="T F"` | One-dimensional boolean array |
| `v="two words"` or `v={two words}` | One string, not a string array |
| `v={3}` | Scalar integer, not a length-one array |
| `v=[1,2.5]` | One-dimensional real array |
| `v=["3","4"]` | String array; quoted bracket elements stay strings |
| `v=[[1,2],[3,4]]` | Two-dimensional integer array |

Whitespace-separated arrays are inferred as a whole: all integers, otherwise all
reals, otherwise all booleans; a non-numeric/non-boolean mixture remains one
string. Unlike scalar real inference, array real inference does not normalise
Fortran exponents. Use ordinary `e`/`E` exponents in arrays.

Bracket arrays are comma-separated. They choose a homogeneous integer, real,
boolean or string kind, in that order; mixed integers and reals promote to real.
If no numeric or boolean kind fits, every element becomes a string. Quoted
commas and brackets inside string elements do not split elements. Only bracket
syntax produces string arrays.

Nested brackets support rectangular two-dimensional arrays, including a single
row or column without squeezing its dimensions. Ragged rows, dimensions above
two, missing row separators and empty elements (including trailing commas) are
errors. Numeric and boolean metadata arrays become NumPy arrays; string arrays
become lists or nested lists.

## Lattice and other conventional names

`Lattice="a1 a2 a3 b1 b2 b3 c1 c2 c3"` is stored as a flat nine-value metadata
array in file order, with the same inference rules as other metadata. Parsing
does not require nine values merely because the key is `Lattice`.
`frame.cell` interprets a valid nine-number value as a `(3, 3)` row-vector cell:
the first three values form the first row. It returns a zero cell if the key is
absent; malformed cells raise a field error when interpreted.

Likewise, native `stress`, `virial`, `pbc` and prefixed training labels retain
their names and represented values. Parsing does not convert virial to stress,
reshape stress or supply missing `pbc`. These interpretations belong to accessors
or [converters](compatibility.md), not the grammar.

## Check inferred values locally

This example distinguishes scalar quoting, explicit string-array elements and
rectangular array shape without an optional converter.

```python title="Check metadata kinds and array shapes"
from pathlib import Path

import numpy as np
import oxyz

path = Path("metadata-types.extxyz")
path.write_text(
    '1\nProperties=label:S:1:flags:L:2 step=" 012 " enabled=t '
    'note=" padded " tags=["3","4"] grid=[[1,2],[3.5,4]] '
    "literal='hello'\nsite_A 1 F\n"
)
frame = oxyz.read(path, 0)
assert frame.columns["label"] == ["site_A"]
assert frame.columns["flags"].shape == (1, 2)
assert frame.columns["flags"].tolist() == [[True, False]]
assert type(frame.metadata["step"]) is int
assert frame.metadata["step"] == 12
assert frame.metadata["enabled"] is True
assert frame.metadata["note"] == " padded "
assert frame.metadata["literal"] == "'hello'"
assert frame.metadata["tags"] == ["3", "4"]
assert frame.metadata["grid"].dtype == np.float64
assert frame.metadata["grid"].shape == (2, 2)
assert frame.metadata["grid"].tolist() == [[1.0, 2.0], [3.5, 4.0]]
assert "Properties" not in frame.metadata
```

## Contract sources

- [Parser](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/extxyz.rs)
  and [native model](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/src/model.rs).
- [Grammar conformance](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/grammar_conformance.rs),
  based on a pinned libAtoms grammar example set, not a promise of every dialect.
- [Invalid corpus](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/corpus_invalid.rs)
  and [valid corpus scoreboard](https://github.com/theochemtheo/oxyz/blob/main/crates/oxyz-core/tests/corpus_smoke.rs).
- [Python value and shape contracts](https://github.com/theochemtheo/oxyz/blob/main/tests/test_extxyz.py).

See the [API reference](api.md), [writing guide](../guides/writing.md) and
[lossless-core decision](../development/decisions/lossless-core-and-layering.md).
