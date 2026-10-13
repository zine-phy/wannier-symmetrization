# Magnetic collinear NSOC: separate up/down HRs

Use for collinear `ISPIN=2` calculations without SOC or noncollinearity. Both scalar channels must share the same atomic-orbital gauge, site order, local axes, home cells and energy zero. Equal dimensions or similar bands do not establish a common gauge.

Add to [the common schema](input-schema.md): `magnetic=true`, `spinor=false`, `spin_order="separate-up-down"`, `moments_muB` (one signed scalar per structural atom), and explicit Boolean `real_gauge`. Include `evidence.common_gauge` and `evidence.real_gauge`. In a verified real-orbital gauge, `real_gauge=true` enforces channel-wise real hopping. Do not enable it to conceal an unaudited gauge.

The algorithm obtains spatial candidates from the full chemical structure. A candidate is preserving if every mapped moment equals the original moment within tolerance; it exchanges channels if every mapped moment equals the negative original moment. Magnitudes and nonprojected atoms participate. Projected shells must also transform consistently. This is a collinear spin-space operation selection, not the SOC axial-vector MSG rule.

All preserving operations are averaged in each channel. If exchanging operations exist, they form a coset: select one representative and average its action with the preserving-subgroup result. Closure checks enforce the construction. `--exchange-operation` selects the zero-based operation index recorded in `operations.json`; otherwise the first exchanging operation is used. The code validates against all selected operations, including the unchosen exchanging operations. If none exist, up/down are projected independently; no equality or time-reversal relation is imposed between them. This covers ferromagnets and ferrimagnets whose actual moment patterns admit no exchange symmetry.

```bash
python3 scripts/symmetrize_collinear_hr.py --up-hr demo/up_hr.dat --down-hr demo/down_hr.dat --case-json demo/collinear.json --ws-mode raw --out demo/collinear_result
```

For WS interpolation use `--ws-mode apply` and both matching `--up-wsvec` and `--down-wsvec`. Optional `native_bands` (a two-element list in up/down order) entries contain `kpt`, `dat` and `tolerance_eV` to check the original Wannier90 interpolation. `--prepare-only` permits inspecting preparation and operations without full projection.

Missing R blocks are zero-extended on a symmetry-closed support. Outputs include `wannier90_up_hr_sym.dat`, `wannier90_down_hr_sym.dat`, prepared/original files, operations, manifest, validation, spectra and a two-channel band overlay. Inspect each channel's before/after bands with one common energy reference, and verify exchange splitting is retained where symmetry permits it. Real-material AFM and synthetic FM tests exist; there is no dedicated real-material ferrimagnetic acceptance benchmark.
