# Nonmagnetic scalar and SOC workflows

Use only for a confirmed nonmagnetic model. Follow [the common schema](input-schema.md) and [the basis audit](basis-and-axes.md).

| Case | JSON fields | Projection |
| --- | --- | --- |
| Scalar NSOC | `magnetic=false`, `spinor=false`, `spin_order="scalar"` | Spatial group and spinless time reversal K in the verified real-orbital gauge |
| SOC | `magnetic=false`, `spinor=true`, `spin_order="all-up-then-all-down"` | Orbital and spin representations of the spatial group, together with spinful time reversal |

The scalar branch enforces real hopping in this gauge. This is not a rule for arbitrary complex Wannier gauges. The SOC branch retains complex hopping, including purely imaginary R blocks; it uses the full complex magnitude when dropping negligible blocks. Space-group symmetry plus time reversal gives the nonmagnetic grey-group constraints. SOC does not imply magnetism.

Generate the synthetic examples as described in [setup](setup.md), then run from the skill root:

```bash
python3 scripts/symmetrize_hr.py --hr demo/scalar_hr.dat --case-json demo/scalar.json --ws-mode raw --out demo/scalar_result
python3 scripts/symmetrize_hr.py --hr demo/soc_hr.dat --case-json demo/soc.json --ws-mode raw --out demo/soc_result
```

For a real calculation, replace the HR and JSON with audited inputs. `--ws-mode apply --wsvec INPUT_WS.dat` absorbs matching WS corrections before projection. `--prepare-only` writes prepared inputs without claiming completed symmetrization. See `--help` for tolerances.

The runner writes original/prepared HRs, core input files, `wannier90_hr_sym.dat`, operation representations, manifest, validation JSON, generic/path spectra and `bands_before_after.{png,pdf,svg}`. It checks delivered-file symmetry, Hermiticity, repeat projection and, for SOC, Kramers pairs at eight TRIMs. Inspect the final band overlay and maximum/RMS differences. Missing R support is completed by symmetry-related atom-pair blocks before averaging; this differs from the magnetic zero-extension policy.
