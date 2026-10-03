"""Notebook rendering: `_repr_html_` tables with a bounded preview of values.

Parsed back with `html.parser`, so the assertions read cell text rather than
markup. The contract: 3 preview rows, arrays in full up to 12 elements, cells
cut at 80 characters, floats to 4 decimal places independent of numpy's global
printoptions, and every name and value escaped.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import oxyz
from oxyz import ColumnRule, Frame, FrameRule, Kind, MetadataRule, SchemaSpec

if TYPE_CHECKING:
    from oxyz import Mode

DATA_DIR = Path(__file__).parent / "data"
PERIODIC = DATA_DIR / "minimal_periodic.extxyz"  # 2 Si, float Lattice
VARYING = DATA_DIR / "varying_atom_counts.xyz"  # 3, 1, 2 atoms; forces, energy
NEWSTYLE = DATA_DIR / "newstyle_array_metadata.extxyz"  # pbc, int/str/float arrays
DRIFT = DATA_DIR / "schema_drift_type.extxyz"
OPTIONAL = DATA_DIR / "mixed_schema_optional_column.xyz"


class Rendered(HTMLParser):
    """Title lines and captioned tables of cell text, header rows dropped."""

    def __init__(self, markup: str) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.tables: dict[str, list[list[str]]] = {}
        self.headers: dict[str, list[str]] = {}
        self._text: list[str] | None = None
        self._caption = ""
        self._row: list[str] | None = None
        self._in_head = False
        self.feed(markup)

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag in ("p", "caption", "td", "th"):
            self._text = []
        elif tag == "tr":
            self._row = []
        elif tag == "thead":
            self._in_head = True

    def handle_endtag(self, tag: str) -> None:
        text = "".join(self._text or [])
        if tag == "p":
            self.lines.append(text)
        elif tag == "caption":
            self._caption = text
            self.tables[text] = []
        elif tag in ("td", "th") and self._row is not None:
            self._row.append(text)
        elif tag == "tr" and self._row is not None:
            if self._in_head:
                self.headers[self._caption] = self._row
            else:
                self.tables[self._caption].append(self._row)
            self._row = None
        elif tag == "thead":
            self._in_head = False
        if tag in ("p", "caption", "td", "th"):
            self._text = None

    def handle_data(self, data: str) -> None:
        if self._text is not None:
            self._text.append(data)


def render(obj: object) -> Rendered:
    return Rendered(obj._repr_html_())  # ty: ignore[unresolved-attribute]


def test_frame_lists_columns_and_metadata_with_values() -> None:
    page = render(oxyz.read(PERIODIC, 0))

    assert page.lines == ["Frame · 2 atoms"]
    assert page.headers["columns"] == ["name", "dtype", "shape", "preview"]
    assert page.tables["columns"] == [
        ["species", "str", "(2,)", "'Si', 'Si'"],
        ["pos", "float64", "(2, 3)", "[0.0 0.0 0.0], [1.5 1.5 1.5]"],
    ]
    assert page.headers["metadata"] == ["name", "dtype", "shape", "value"]
    assert page.tables["metadata"] == [
        ["Lattice", "float64", "(9,)", "[3.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 3.0]"],
    ]


def test_frame_preview_shows_three_rows_then_elides() -> None:
    first = render(oxyz.read(VARYING, 0))
    batch = render(oxyz.read_batch(VARYING))

    assert first.tables["columns"][1] == [
        "pos",
        "float64",
        "(3, 3)",
        "[0.0 0.0 0.0], [0.758 0.0 0.504], [-0.758 0.0 0.504]",
    ]
    assert batch.tables["columns"][1][3] == (
        "[0.0 0.0 0.0], [0.758 0.0 0.504], [-0.758 0.0 0.504], …"
    )
    assert first.tables["metadata"] == [
        ["Lattice", "int64", "(9,)", "[10 0 0 0 10 0 0 0 10]"],
        ["energy", "float", "()", "-76.3"],
    ]


def test_frame_renders_bool_int_and_string_array_metadata() -> None:
    page = render(oxyz.read(NEWSTYLE, 0))
    rows = {row[0]: row[1:] for row in page.tables["metadata"]}

    assert rows["pbc"] == ["bool", "(3,)", "[True True False]"]
    assert rows["kpoints"] == ["int64", "(3,)", "[2 2 1]"]
    assert rows["tags"] == ["str", "(2,)", "['slab' 'relaxed']"]
    assert rows["cutoffs"] == ["float64", "(2,)", "[4.5 5.0]"]


def oddments(**metadata: object) -> Frame:
    return Frame(
        n_atoms=1,
        columns={"species": ["H"], "pos": np.zeros((1, 3))},
        metadata=metadata,  # ty: ignore[invalid-argument-type]
    )


def test_a_long_array_shows_its_first_twelve_elements() -> None:
    page = render(
        oddments(
            flat=np.arange(20.0), grid=np.arange(20).reshape(5, 4), short=np.arange(12)
        )
    )
    rows = {row[0]: row[3] for row in page.tables["metadata"]}

    assert rows["flat"] == ("[0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 …]")
    assert rows["grid"] == "[[0 1 2 3] [4 5 6 7] [8 9 10 11] …]"
    assert rows["short"] == "[0 1 2 3 4 5 6 7 8 9 10 11]"


def test_a_numpy_string_array_renders_as_plain_strings() -> None:
    page = render(oddments(labels=np.array(["a", "b"])))

    assert page.tables["metadata"] == [["labels", "<U1", "(2,)", "['a' 'b']"]]


def test_a_long_cell_is_cut_at_eighty_characters() -> None:
    page = render(oddments(note="x" * 200))
    [[_, _, _, value]] = page.tables["metadata"]

    assert len(value) == 80
    assert value == "'" + "x" * 78 + "…"


def test_floats_round_to_four_places_and_ignore_global_printoptions() -> None:
    frame = oddments(energy=-411.209345678, small=1e-9, ratio=2 / 3)
    before = frame._repr_html_()

    with np.printoptions(precision=1, floatmode="fixed", suppress=False):
        assert frame._repr_html_() == before
    rows = {row[0]: row[3] for row in Rendered(before).tables["metadata"]}
    assert rows == {"energy": "-411.2093", "small": "0.0", "ratio": "0.6667"}


def test_names_and_values_are_escaped() -> None:
    hostile = Frame(
        n_atoms=1,
        columns={"species": ["<b>H</b>"], "<script>x</script>": np.zeros(1)},
        metadata={"note": "</td><td>injected"},
    )
    markup = hostile._repr_html_()
    page = Rendered(markup)

    assert "<script>" not in markup
    assert "<b>" not in markup
    assert page.tables["columns"][1][0] == "<script>x</script>"
    assert page.tables["columns"][0][3] == "'<b>H</b>'"
    assert page.tables["metadata"] == [["note", "str", "()", "'</td><td>injected'"]]


def test_the_title_is_escaped_too() -> None:
    # SchemaSpec does not validate `mode` on construction.
    markup = SchemaSpec(mode=cast("Mode", "<i>x</i>"))._repr_html_()

    assert "<i>" not in markup
    assert Rendered(markup).lines == ["SchemaSpec · <i>x</i>"]


def test_a_huge_frame_renders_a_small_page() -> None:
    n = 100_000
    huge = Frame(
        n_atoms=n,
        columns={"species": ["H"] * n, "pos": np.zeros((n, 3))},
        metadata={"Lattice": np.eye(3).ravel()},
    )

    assert len(huge._repr_html_()) < 3000


def test_values_it_cannot_format_render_as_their_type() -> None:
    odd = Frame(
        n_atoms=2,
        columns={"ragged": [["a"], ["b", "c"]]},
        metadata={"things": np.array([object(), None], dtype=object)},
    )
    page = render(odd)

    assert page.tables["columns"][0][0] == "ragged"
    assert page.tables["metadata"][0][0] == "things"


def test_batch_titles_its_layout_and_previews_per_frame_metadata() -> None:
    page = render(oxyz.read_batch(VARYING))

    assert page.lines == ["Batch · 3 frames · 6 atoms (1..3 per frame)"]
    rows = {row[0]: row[1:] for row in page.tables["metadata"]}
    assert rows["energy"] == ["float64", "(3,)", "-76.3, -13.6, -31.8"]
    assert rows["Lattice"] == [
        "int64",
        "(3, 9)",
        "[10 0 0 0 10 0 0 0 10], [10 0 0 0 10 0 0 0 10], [10 0 0 0 10 0 0 0 10]",
    ]


def test_schema_tabulates_types_presence_and_unification() -> None:
    drift = render(oxyz.infer_schema(DRIFT))
    optional = render(oxyz.infer_schema(OPTIONAL))

    assert drift.lines == [
        "Schema · 2 frames · 4 atoms (2..2 per frame) · inconsistent"
    ]
    assert drift.headers["columns"] == ["name", "type", "frames", "unified"]
    assert drift.tables["columns"] == [
        ["species", "S:1", "2/2", "S:1"],
        ["pos", "R:3", "2/2", "R:3"],
        ["magmom", "R:3|R:1", "2/2", "—"],
    ]
    assert drift.tables["metadata"] == [["energy", "R", "2/2", "R"]]
    assert optional.tables["columns"][2] == ["charge", "R:1", "1/2", "R:1"]


def test_spec_tabulates_its_rules_with_unset_fields_blank() -> None:
    spec = SchemaSpec(
        columns=(
            ColumnRule("pos", Kind.REAL, width=3),
            ColumnRule("d_*", Kind.REAL, count=5),
            ColumnRule("tag", Kind.INT, required=False, fill=0),
        ),
        metadata=(MetadataRule("Lattice", Kind.REAL, shape=(9,), required=False),),
        frame=FrameRule(n_atoms_max=500),
        mode="project",
    )
    page = render(spec)

    assert page.lines == ["SchemaSpec · project", "frame: FrameRule(n_atoms_max=500)"]
    assert page.headers["columns"] == [
        "name",
        "type",
        "required",
        "count",
        "min",
        "max",
        "fill",
    ]
    assert page.tables["columns"] == [
        ["pos", "R:3", "yes", "", "", "", ""],
        ["d_*", "R:1", "yes", "5", "", "", ""],
        ["tag", "I:1", "no", "", "", "", "0"],
    ]
    assert page.tables["metadata"] == [["Lattice", "R[9]", "no", "", "", "", ""]]


def test_frame_index_tabulates_atom_count_and_volume_statistics() -> None:
    page = render(oxyz.scan(VARYING, with_volume=True))

    assert page.lines == ["FrameIndex · 3 frames · 6 atoms"]
    assert page.tables["atoms per frame"] == [
        ["min", "1"],
        ["median", "2.0"],
        ["mean", "2.0"],
        ["max", "3"],
        ["std", "0.8165"],
    ]
    assert page.tables["volume"] == [
        ["min", "1000.0"],
        ["mean", "1000.0"],
        ["max", "1000.0"],
        ["frames without", "0"],
    ]
    assert "volume" not in render(oxyz.scan(VARYING)).tables


def test_volume_statistics_skip_frames_without_a_lattice(tmp_path: Path) -> None:
    mixed = tmp_path / "mixed.extxyz"
    mixed.write_text(
        '1\nLattice="2 0 0 0 2 0 0 0 2" Properties=species:S:1:pos:R:3\nH 0 0 0\n'
        "1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n"
        '1\nLattice="4 0 0 0 4 0 0 0 4" Properties=species:S:1:pos:R:3\nH 0 0 0\n'
    )
    page = render(oxyz.scan(mixed, with_volume=True))

    assert page.tables["volume"] == [
        ["min", "8.0"],
        ["mean", "36.0"],
        ["max", "64.0"],
        ["frames without", "1"],
    ]


def test_frame_index_of_an_empty_file_shows_dashes(tmp_path: Path) -> None:
    empty = tmp_path / "empty.xyz"
    empty.touch()
    page = render(oxyz.scan(empty))

    assert page.lines == ["FrameIndex · 0 frames · 0 atoms"]
    assert {value for _, value in page.tables["atoms per frame"]} == {"—"}


@pytest.mark.parametrize(
    "obj",
    [
        oxyz.read(PERIODIC, 0),
        oxyz.read_batch(VARYING),
        oxyz.infer_schema(VARYING),
        oxyz.infer_schema(VARYING).to_spec(),
        oxyz.scan(VARYING),
    ],
    ids=["Frame", "Batch", "Schema", "SchemaSpec", "FrameIndex"],
)
def test_every_rendering_is_one_plain_div(obj: object) -> None:
    markup = obj._repr_html_()  # ty: ignore[unresolved-attribute]

    assert markup.startswith('<div class="oxyz-repr">')
    assert markup.endswith("</div>")
    for banned in ("<script", "<style", "style=", "color"):
        assert banned not in markup
