# Magnetic spinor MSG workflow (experimental)

This branch symmetrizes a complete spinor HR using the magnetic space group compatible with a supplied magnetic structure. It supports unitary and antiunitary operations and does not impose pure time reversal on an ordered magnet. The numerical implementation has analytic tests and material benchmarks; the workflow remains experimental because the source gauge and magnetic structure require a case-specific audit.

Add to [the common schema](input-schema.md):

- `magnetic=true`, `spinor=true`, `target_symmetry="MSG"`.
- `source_soc`: actual Boolean source setting, not the desired test label.
- `spin_order`: `"interleaved"` or `"all-up-then-all-down"`.
- `spin_frame="global-cartesian"`; spin basis and moments must already be expressed in this frame.
- `moments_cartesian_muB`: one Cartesian three-vector per structural atom.
- Additional `evidence` keys: `spin_frame`, `atomic_gauge`, `moments`.

For Cartesian spatial operation O and atom map i to j, a unitary operation satisfies m_j = det(O) O m_i; an antiunitary operation satisfies m_j = -det(O) O m_i, within the moment tolerance. Translations and atom home-cell shifts enter hopping transformations. Local orbital representations combine with the proper spin rotation. Antiunitary actions include spin time reversal and complex conjugation of the Hamiltonian. The projector averages the complete accepted MSG on a symmetry-closed, zero-extended R support.

```bash
python3 scripts/symmetrize_magnetic_hr.py --hr demo/magnetic_hr.dat --case-json demo/magnetic.json --ws-mode raw --out demo/magnetic_result
```

Use matching `--wsvec` with `--ws-mode apply` when required. Optional `native_bands` has `kpt`, `dat`, `tolerance_eV`. Output includes `wannier90_hr_sym.dat`, prepared/original inputs, operations, manifest, validation, spectra and the mandatory band overlay. Validate both unitary/antiunitary residuals, Hermiticity, repeat projection, Bloch covariance and changes in the full spinor bands. Pure-TR difference is diagnostic and may remain nonzero.

For magnetic noncollinear NSOC, use the [SSG workflow](ssg.md), which allows independent spin rotations and includes the coplanar spin-only mirror. The MSG runner still supports explicitly requested subgroup comparisons with `source_soc=false`; passing such a test does not establish full SSG invariance or an acceptable DFT fit. The original Mn3Sn MSG benchmark is only such a subgroup test; the official Co-S spinor example is a separate MSG benchmark. See [validation](validation.md).
