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
| `molecule_type_labels.extxyz` | Per-atom *string* column (`molecule_type:S:1`) — column types beyond numbers. |
| `mace_ref_energy_forces_stress.xyz` | MACE training schema: `REF_energy`, `REF_forces:R:3`, `REF_stress`, `config_type`. |
| `mace_isolated_atom_and_head.xyz` | MACE isolated-atom frames; per-frame `config_type`/`head` metadata that differs by frame. |
