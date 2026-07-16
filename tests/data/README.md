# extxyz fixture corpus

| File | Exercises |
|---|---|
| `simple.extxyz` | Original spike fixture: full required-field set, 1 H atom. |
| `nonorthogonal.extxyz` | Non-orthogonal cell (Fortran-order `Lattice` → row-major), mixed `pbc`. |
| `minimal_periodic.extxyz` | Smallest periodic file: `species` + `pos` only, no energy/forces/stress/pbc. |
| `periodic_pbc_ttf.extxyz` | Slab-style explicit `pbc="T T F"` without other frame data. |
| `no_lattice_molecule.xyz` | `Properties` but no `Lattice` (valid per libAtoms spec); frame-level string/int metadata. |
| `atomic_numbers_z.extxyz` | `Z:I:1` instead of `species:S:1`; atom identity from integers. |
| `mass_and_charge.extxyz` | Extra scalar real per-atom columns; no `Lattice`. |
| `id_and_selection.extxyz` | Integer per-atom columns; species not in first column. |
| `force_singular_alias.extxyz` | Singular `force:R:3` alias for `forces`. |
| `quoted_strings_booleans_scalars.extxyz` | Comment-line value typing: quoted string with spaces, bare string, `T`/`F` booleans, int, real. |
| `singlequote_metadata.extxyz` | Single quotes are ordinary bare characters per the grammar (only `"` quotes); oxyz keeps them where ASE strips them. |
| `newstyle_array_metadata.extxyz` | New-style bracket arrays (`[2,2,1]`, string arrays) in metadata. |
| `stress_voigt6.extxyz` | Frame-level 6-component (Voigt) `stress`. |
| `stress_matrix9.extxyz` | Frame-level 9-component `stress` — same key, different shape than Voigt. |
| `virial_matrix9.extxyz` | Frame-level `virial` (preserve losslessly; no sign/unit conversion at parse time). |
| `two_frame_same_schema.xyz` | Two-frame trajectory, stable schema, per-frame `energy`/`Time`. |
| `varying_atom_counts.xyz` | Three frames with 3/1/2 atoms — forces the ragged-data decision. |
| `varying_density.extxyz` | Three 2-atom frames, one dense cell and two sparse — distinguishes `n_atoms_x_density` binning from plain atom-count binning. |
| `molecule_type_labels.extxyz` | Per-atom *string* column (`molecule_type:S:1`) — column types beyond numbers. |
| `per_atom_boolean.extxyz` | Per-atom *boolean* column (`active:L:1`) — the extxyz `L` kind. |
| `move_mask_fix_cartesian.extxyz` | Per-atom `move_mask:L:3` — ASE writes/reads it as `FixCartesian` constraints. |
| `move_mask_fix_atoms.extxyz` | Per-atom `move_mask:L:1` — ASE writes/reads it as a `FixAtoms` constraint. |
| `mace_ref_energy_forces_stress.xyz` | MACE training schema: `REF_energy`, `REF_forces:R:3`, `REF_stress`, `config_type`. |
| `mace_isolated_atom_and_head.xyz` | MACE isolated-atom frames; per-frame `config_type`/`head` metadata that differs by frame. |
| `schema_conformant.extxyz` | Two frames, stable `species`/`pos`/`energy` schema — the schema-aware read happy path. |
| `schema_extra_column.extxyz` | Second frame adds a `charge` column — exercises `strict` (error) vs `required` (allowed). |
| `schema_drift_type.extxyz` | `magmom` changes width between frames (non-collinear `R:3` → collinear `R:1`) — a realistic per-atom width mismatch at frame 1. |
| `mixed_schema_optional_column.xyz` | Two frames; the second lacks `charge` — the mixed-schema case that projection (`mode="project"`) makes batchable. |
| `mad_r2scan_sample.extxyz` | MAD-1.5 r²SCAN slice: real, chemically diverse data (102 elements; molecules, clusters, bulk, surfaces, low-dimensional) in one standardised DFT workflow. See the source and attribution below. |

Compressed twins of `two_frame_same_schema.xyz` (gzip, zstd, zip, tar, tar.gz,
plus concat-gzip and multi-member archives) live in `compressed/`; see its
README.

## `invalid/` — malformed fixtures

Deterministic regression corpus of files that must fail to parse. Each is
guarded by `corpus_invalid.rs`, which asserts the frame index, the line/column,
and a wording substring of the error — not merely that one was raised.

| File | Malformation |
|---|---|
| `dangling_metadata_value.extxyz` | Comment-line key with no value (`bad=`). |
| `unterminated_quote.extxyz` | Quoted metadata value with no closing `"`. |
| `unknown_property_kind.extxyz` | `Properties` kind letter outside `S`/`I`/`R`/`L`. |
| `short_atom_row.extxyz` | Atom row with fewer columns than the schema declares. |
| `bad_atom_value.extxyz` | Non-numeric value in a real column. |
| `bad_atom_count.extxyz` | Count line that is not a non-negative integer. |
| `truncated_frame.extxyz` | Second frame declares more atoms than the file supplies before EOF. |
| `ragged_bracket_array.extxyz` | 2-D bracket array (`Lattice=[[1,2],[3]]`) whose rows disagree in length. |
| `trailing_comma_array.extxyz` | New-style array with a trailing comma (`tags=[a,b,]`). |
| `bare_string_excluded_char.extxyz` | Bare (unquoted) metadata value containing a grammar-reserved character (`note=a=b`). |

## `mad_r2scan_sample.extxyz` — source and attribution

Sliced from `mad-1.5-r2scan-train.xyz` (303.5 MiB) in *High-quality,
high-information datasets for universal atomistic machine learning* by Cesare
Malosso, Filippo Bigi, Paolo Pegolo, Joseph W. Abbott, Philip Loche, Mariana
Rossi, Michele Ceriotti and Arslan Mazitov — Materials Cloud Archive,
[doi:10.24435/materialscloud:ak-4p](https://doi.org/10.24435/materialscloud:ak-4p)
(v3), described in [arXiv:2603.02089](https://arxiv.org/abs/2603.02089). Used
under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

Reproduce with `scripts/slice_mad_sample.py`: every 900th frame, taken at the
byte level so the sample keeps the source's exact formatting — 201 frames,
381404 bytes, spanning all 14 subsets and 44 distinct atom counts (1–198). The
full dataset is not committed.
