# Automatic POSCAR interface

Use `scripts/symmetrize.py` for new calculations. It constructs the internal case, legacy core inputs and band path. The branch-specific runners remain available for existing audited case files.

## Input contract

Required: a VASP5 POSCAR with species names, the calculation type, and the matching HR (two HRs for collinear magnetic NSOC). Append Cartesian magnetic moments in muB to atom rows:

```text
Direct
0.5 0.5 0.5   1 2 3
0.0 0.0 0.0
```

The first three numbers follow the POSCAR coordinate mode. The last three always mean Cartesian moment components, here `(1, 2, 3)` muB. An omitted triple means zero moment. With Selective dynamics, put the moment triple after the three T/F flags. Lattice scaling changes positions, never moments. Positive scale, negative target volume and three positive Cartesian scale factors are supported. This extended file is an input to this toolkit; do not assume another program reads the extra columns as magnetic moments.

For collinear NSOC, moments must lie on the up/down quantization axis. The default is source SAXIS when present, otherwise Cartesian +z; select another with `--spin-axis x y z`. FM and ferrimagnetic patterns do not require a channel-exchanging operation. Rank-two coplanar and rank-three noncoplanar textures use the noncollinear SSG branch when SOC is disabled. A coplanar noncollinear texture is not an ISPIN=2 model.

| Calculation | Flags | HR arguments |
| --- | --- | --- |
| Nonmagnetic NSOC | `--magnetism nonmagnetic --ordering collinear --soc no` | `--hr` |
| Nonmagnetic SOC | `--magnetism nonmagnetic --ordering collinear --soc yes` | `--hr` |
| Magnetic collinear NSOC | `--magnetism magnetic --ordering collinear --soc no` | `--up-hr`, `--down-hr` |
| Magnetic SOC | `--magnetism magnetic --ordering collinear` or `noncollinear`, `--soc yes` | `--hr` |
| Magnetic noncollinear NSOC | `--magnetism magnetic --ordering noncollinear --soc no` | `--hr` |

`collinear` is only a CLI placeholder for nonmagnetic models. It does not add magnetic order. Magnetic declarations are checked against the moment rank and available source settings; the program does not idealize moments.

## Basis metadata cannot be inferred from HR alone

POSCAR and HR do not encode which orbital each matrix index represents, its phase, local axes, spin frame or gauge. The interface searches the HR directory for matching seed `.win` and `.wout` files, then conventional `wannier90` names; ambiguous matches need `--win` or `--wout`. It optionally reads `OUTCAR` or `OUTCAR.scf` and records VASP/Wannier90 versions, or explicitly marks them unknown. Supply another source OUTCAR with `--outcar`.

WOUT ordered projection rows establish complete s/p/d/f site blocks and duplicated spin ordering. Final WF centers establish consistent integer home-cell shifts; intra-cell displacement is not substituted for an atomic position. Final centers can diagnose, but cannot prove, the correct atomic gauge. Matching up/down WOUT files are cross-checked when both exist. Their labels alone do not certify the actual up/down file assignment or equality of orbital phases.

WIN `num_iter=0` permits the projection-derived atomic-gauge workflow; this remains a physical input contract, not a proof that the disentangled subspace carries exact atomic representations. Otherwise, `--atomic-gauge` records an explicit audit assertion. It does not transform an arbitrary MLWF gauge. Review before/after bands and the original DFT fit before accepting a model.

For nonidentity axes, choose `--axes-convention physical` or `transpose` after checking the actual DFT-to-Wannier interface. No VASP-version-only transpose rule is assumed. Read [basis and axes](basis-and-axes.md). Nondefault source SAXIS and custom projection spin axes require a prior spin-frame conversion for spinor workflows.

Fallbacks when metadata are incomplete:

- A matching WIN projection block supports species or `f=x,y,z` selectors and s/p/d/f/t2g shells. Confirm `--home-cells poscar`; spinors also require `--spin-order interleaved` or `all-up-then-all-down` when no WOUT establishes the order.
- Explicit `--orbitals 'Mn:d;Sn:p'` follows the stated species/site order in Cartesian orbital axes. Use one-based site selectors such as `'1:d;2:p'` for site-specific order. Confirm `--home-cells poscar` and, if needed, `--atomic-gauge` and spin ordering. This override is an assertion, not an orbital-order guess.
- `--basis-json` imports an existing audited basis, local axes and home cells. Its optional lattice/atom metadata must match POSCAR. The interface still generates the new case and magnetic operations.

Unsupported projection syntax, repeated radial shells, interleaved atom blocks or ambiguous centers require an audited basis/permutation rather than an automatic guess. Explicit projected spin directions in WIN are rejected by this interface. See the [official projection definitions](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/projections/) for upstream syntax; this parser deliberately supports a subset.

## Interpolation, path and output

`--ws-mode auto` follows WIN `use_ws_distance` and matching seed `_wsvec.dat` files. An explicit true setting without required WS data fails. When both settings and WS data are absent, it records the bare-HR Fourier convention. Use `--ws-mode raw` or `apply` to choose explicitly; paired apply mode needs both channel files. Absorbed WS vectors must not be reapplied to the output HR.

The comparison path comes from matching WIN `kpoint_path`, otherwise [SeeK-path](https://seekpath.readthedocs.io/en/latest/maindoc.html) supplies a chemical-structure path expressed in the original reciprocal basis. For supercells, its labels refer to the associated primitive-cell BZ; they are not magnetic little-group labels. No input cell is transformed. `--points` controls samples per segment. Use `--fermi` for a common energy zero; otherwise WIN `fermi_energy` is used when available, and absent Fermi metadata leave the HR energy zero unchanged.

A fresh `--out` directory contains:

- `case.json`, `command.json`, `input_provenance.json`: generated inputs, exact command, source hashes, conventions and completion status.
- `wannier90_hr_sym.dat`, or `wannier90_up_hr_sym.dat` and `wannier90_down_hr_sym.dat`: delivered models in the original external spin order. Output R degeneracies are one.
- `result/operations.json`: rotations/translations, atom maps, home shifts, orbital/spin matrices and unitary/antiunitary or channel-preserving/exchanging labels. Nonmagnetic exports include spatial operations and their time-reversed partners. Matrices use the documented internal order; delivered HR order is recorded separately.
- `result/validation.json`, `manifest.json`, `bands.npz`, and `bands_before_after.png/.pdf/.svg`: numerical checks, common-path spectra and overlay. Band differences pair sorted eigenvalues, without eigenvector tracking.

`--prepare-only` constructs inputs without projecting HR. A completed projection always produces before/after bands. Source files are never overwritten and their hashes are checked again after execution.
