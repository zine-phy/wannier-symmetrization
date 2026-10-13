# Basis, local axes and source-version audit

An HR file alone does not identify its orbitals, local axes, spin ordering or home cells. Recover matching structure, projection definitions, actual Wannier interface behavior and gauge provenance. Record VASP/Wannier90 versions and, when available, source/build identifiers; mark missing versions unknown rather than inferring them from file dates.

## Orbital representation

The real-harmonic shell order in the bundled rotation module is:

| Shell | Order |
| --- | --- |
| s | s |
| p | pz, px, py |
| d | dz2, dxz, dyz, dx2-y2, dxy |
| t2g | dxz, dyz, dxy |
| f | z(2z²-3x²-3y²), x(4z²-x²-y²), y(4z²-x²-y²), z(x²-y²), xyz, x(x²-3y²), y(3x²-y²), with module normalization |

Each site's axes matrix A has physical Cartesian x, y, z axes as columns, with y = z cross x. `axes_convention="physical-cartesian-columns"` explicitly declares this. Incomplete shells such as t2g must be closed under every retained operation; otherwise use a closed basis or establish a justified transformation.

Orbital axes and spin axes are separate. The magnetic spinor branch currently requires a global Cartesian spin frame. Transform a source spin frame into it before running. Independent Wannier optimizations can mix orbitals and introduce phases; writing different axes in JSON does not perform the necessary HR gauge transformation.

## Audited VASP interface difference

For A formed from the `.win` projection axes, source inspection of specific VASP builds found different coefficient contractions:

| Inspected build | Actual physical orbital axes in the audited interface path |
| --- | --- |
| 6.1.1 | A |
| 6.4.1 | A transpose |

Both constructed the coordinate transformation U = A transpose; the distinction arose downstream from contracting rows versus columns of the harmonic rotation matrix. This is a source-build audit, not a documented universal rule for every VASP release or interface path. Identity axes conceal the difference. Do not select a transpose from the version number alone; confirm the actual source path or a suitable orbital-projection check.

Audit fingerprints (SHA-256; proprietary source is not bundled):

- 6.1.1 `mlwf.F`: `7d568483df04cc473ee4c811984f77b8e9e7bb2202420decc8bfb6c0ffc88b0f`.
- 6.4.1 `locproj.F`: `f87b4c2f433e2d8be10011b1bcb74fea152aea7c571f873d51588ac28bd9d0c2`.
- 6.4.1 `dfast.F`: `ae0ac7b4660b861554cf1f9da5daf8264b2bb0b0dcc3526a65309006028c3841`.
- 6.4.1 `mlwf.F`: `7ef49e4c04f5170d69ea7526d040f3594c1a6116f80f6dd645718d88a7e8d180`.

The inspected 6.4.1 interface also folded projection positions into [-0.5, 0.5); establish the model's actual integer home cells instead of copying POSCAR representatives blindly. Changing a home cell changes R-dependent phases. Fractional Wannier-center drift is not an integer home-cell correction. File suffixes for separate spin channels may differ between builds; identify each channel from actual calculation output.

For convention background consult the [Wannier90 projection documentation](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/projections/) and [VASP LWANNIER90 documentation](https://www.vasp.at/wiki/index.php/LWANNIER90). These links do not establish the build-specific transpose finding above.
