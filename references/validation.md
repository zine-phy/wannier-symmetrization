# Validation scope

The distributable tests are synthetic and require no private files. They cover weighted HR reading, arbitrary R ordering, purely imaginary blocks, spin ordering, spinless/spinful TR, local orbital representations, subgroup/coset projection, AFM/FM operation selection, antiunitary MSG action, independent Bloch-space projection and relocation of the complete CLI workflows.

The current distribution passes 43 bundled tests, including numerical regressions, five relocated low-level CLI workflows, missing-band-path rejection, automatic POSCAR/metadata parsing, all five automatic workflows and restored interleaved SOC output. This release was tested on macOS ARM64 with Python 3.12.14, NumPy 2.5.4, SymPy 1.14.0, SciPy 1.18.1, spglib 2.8.0, Matplotlib 3.11.2 and SeeK-path 2.2.2. The previous 32-test numerical/relocation suite also passed on Linux with NumPy 1.26.4 and spglib 2.6.0. An optional GitHub Actions template defines Linux, macOS and Windows jobs; it is not activated in this publication. The configured matrix is not a claim of verified compatibility. Dependency ranges describe installation constraints, not an exhaustive compatibility matrix.

Material checks performed during development are summarized below. Input datasets and generated run manifests are intentionally not included. These are recorded benchmark results, not tests automatically rerun by this package.

| Branch / benchmark | Key result | Scope |
| --- | --- | --- |
| NM scalar, 19-orbital V2Se2O | Historical HR difference 8.44e-15 eV; repeat 2.66e-15 eV | Scalar regression |
| NM SOC, 32-orbital TaAs | Symmetry residual 9.93e-16 eV; Kramers splitting 1.78e-14 eV | Nonmagnetic spinor regression |
| NM local-axis model, 19 orbitals | Residual 1.11e-16 eV; path change about 0.376 eV | Demonstrates the importance of physical axes; not a DFT-fit acceptance |
| Collinear KVSeO | 8 preserving + 8 exchanging operations; all representative choices agree within 1.36e-20 eV | Real AFM test; subgroup/coset result agrees with full-group projection within 8.88e-16 eV |
| Collinear KVSeO bands | Up/down maximum changes 4.55e-7 / 8.14e-7 eV; generic-k spin splitting about 0.928 eV retained | Common-gauge, before/after comparison |
| MSG Mn3Sn NSOC spinor | 4 unitary + 4 antiunitary; BNS 63.463; residual 2.70e-15 eV | MSG-subgroup test only; original disentanglement unconverged; path change 0.931 eV is not an accepted DFT fit |
| MSG official Co-S spinor example | 4 unitary + 4 antiunitary; BNS 61.436; residual 1.78e-15 eV; repeat 3.55e-15 eV | Full complex spinor MSG benchmark; source DFT/Wannier versions unavailable |
| Co-S before/after bands | Maximum 8.77306e-4 eV, RMS 4.34359e-5 eV; pure-TR difference about 0.755 eV retained | Raw-HR interpolation; magnetic splitting preserved |

The Co-S benchmark uses the intended spinor/SOC example supplied with upstream; matching OUTCAR settings are not bundled. Numerical agreement with all current symmetry constraints does not imply exact equality to the legacy output, which used different support/projection handling.

Ferromagnetic behavior is tested with synthetic models; there is no dedicated real-material ferrimagnetic acceptance benchmark. Noncollinear NSOC SSG has an experimental implementation with coplanar/noncoplanar synthetic tests and a coplanar Mn3Sn trial; no real-material noncoplanar acceptance benchmark is included. Re-run the bundled tests after dependency changes, and validate each new real model's source representation and final bands.

## Noncollinear NSOC SSG validation

The Mn3Sn operation search agrees with the operation-finding prefix of MOM2SSG `findAllOp` at the source commit documented in [SSG](ssg.md). Upstream returns 24 coplanar representatives; completing them with the spin-only mirror gives the same 48 operations as this implementation (24 unitary, 24 antiunitary), with maximum spin-matrix difference 3.61e-16. SSG numbering and its databases were not used.

The matching INCAR specifies exact 120-degree coplanar directions. These directions were scaled to the mean final Mn moment magnitude to avoid nonclosed rotations from the OUTCAR's rounded moments. Original printed moments were retained; the maximum adjustment was 4.40601e-4 muB. This case-specific, documented template is not automatic normalization performed by the runner.

For the 72-dimensional Mn3Sn HR with matching WS interpolation, original Wannier90 band reproduction error was 1.71e-5 eV. After SSG projection, maximum operation residual was 6.22e-15 eV, repeat projection error 5.33e-15 eV, and Bloch covariance error 1.42e-14 eV. Pure-TR difference remained about 2.611 eV, as expected for the ordered magnet.

The final path bands changed by at most 0.951127 eV (RMS 0.0744183 eV); within 1 eV of the common Fermi level the maximum change was 0.524569 eV. The overlay shows substantial changes around the basal-plane path. The source disentanglement was not converged and its atomic gauge remains an assumption needing review. This is a successful numerical workflow test, not an accepted symmetrized DFT model.
