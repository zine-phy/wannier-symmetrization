# Magnetic noncollinear NSOC: spin-space-group projection

Use `scripts/symmetrize_ssg_hr.py` for a complete non-SOC spinor HR with a verified noncollinear magnetic structure. Collinear models remain in the existing ISPIN=2 workflow; this runner intentionally rejects rank-one and zero-moment inputs. Both coplanar (rank two) and noncoplanar (rank three) textures are supported. No SSG number is needed or computed.

## Operation search and relation to MOM2SSG

The search follows the atom/moment compatibility condition in [MOM2SSG findAllOp](https://github.com/zine-phy/MOM2SSG/blob/2b67cc45d66f81d26ae9af6453e3652b7791f06d/identifySSG/identifySSG.py): for each chemical spatial operation (R,t), identify its species-preserving atom permutation i to j, then retain independent orthogonal spin operations U satisfying m_j = U m_i. All structural atoms and moment magnitudes participate, including atoms outside the Wannier basis. Spatial R is in fractional lattice coordinates; U is in global Cartesian spin coordinates.

The bundled implementation solves the orthogonal map directly by singular-value decomposition for each atom permutation. This replaces upstream's spin-point-group candidate enumeration; it does not import its SSG classification database or require pymatgen. Rank-three moments determine at most one U for each spatial permutation. Rank-two moments determine the action within their plane and allow both signs of the normal extension.

Upstream `findAllOp` calls `twoD2threeD` for a coplanar texture, adding a common normal component to select representatives. For complete Hamiltonian symmetrization, retain both the representatives and their products with the spin-only mirror M = I - 2 n n^T, where n is the unit normal to the moment plane. This mirror fixes every moment in the plane; it is an antiunitary operation on spinors. The new search directly returns both extensions, so do not double them a second time. For a noncoplanar texture there is no extra nontrivial spin-only stabilizer.

## Representation and averaging

The orbital matrix is determined by the spatial operation and the audited local orbital axes, as in MSG. The spin matrix now comes from the independent U, not from det(O) O for the Cartesian spatial operation O.

- det(U)=+1: use its SU(2) lift D(U), a unitary operation.
- det(U)=-1: use D(-U) T, with T=i sigma_y K, an antiunitary operation. Complex conjugation acts on the entire HR matrix.

The Kronecker product of orbital and spin matrices, atom permutation and home-cell shifts defines the action on hopping blocks. The shared magnetic projector averages all accepted operations on a zero-extended symmetry-closed R support, preserves complex hopping, and checks the semilinear double-group representation. Do not impose pure T or real hopping on a magnetic spinor model. The coplanar mirror generally does not make all matrix entries real in the chosen global spin frame.

## Input and execution

Use [the magnetic spinor schema](magnetic.md), except set `target_symmetry="SSG"` and `source_soc=false`. The runner refuses SOC inputs. Keep `spin_frame="global-cartesian"`, verified spin order, physical orbital axes and complete `moments_cartesian_muB`. All evidence fields from the magnetic schema remain required.

```bash
python3 examples/make_demo.py --out demo
python3 scripts/symmetrize_ssg_hr.py --hr demo/ssg_hr.dat --case-json demo/ssg.json --ws-mode raw --out demo/ssg_result
```

Use `--ws-mode apply --wsvec INPUT_WS.dat` for matching WS interpolation. `band_path` is mandatory. The common output includes the symmetrized HR, manifest, original/prepared files, numerical validation, spectra and PNG/PDF/SVG before/after overlays. `operations.json` additionally records independent Cartesian spin rotations, their determinants, moment rank and antiunitary flags.

`--moment-tol` is the maximum atomwise moment-matching error in muB. `--rank-tol` sets the moment-rank threshold: singular values exceeding rank_tol times sqrt(number of atoms) are retained. Neither tolerance licenses silently changing the magnetic order. Rounded moments can produce nonclosed approximate spin rotations even when pairwise errors are small. The code rejects failed group closure. Recover a documented symmetry-consistent moment template from the source calculation, retain the observed moments, and report the adjustment; do not normalize unequal moments or impose coplanarity without evidence.

## Completion and scope

Verify operation closure, moment compatibility, delivered-file unitary/antiunitary covariance, Hermiticity, repeat projection and Bloch covariance. Finish with inspected before/after bands on one common k path, energy reference and interpolation convention. The operation search describes symmetry of the supplied magnetic structure; independent orbital-current order, an incompatible gauge or a poor Wannier fit may invalidate imposing that symmetry on a particular HR. Retain the experimental status until the representation and band changes are acceptable for the target model.
