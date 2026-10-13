# Interpolation and mandatory band comparison

The HR reader divides each matrix block by its recorded R degeneracy exactly once. Output files contain physically weighted matrices and degeneracy weights one.

`--ws-mode raw` means the ordinary Fourier sum H(k) = sum_R H(R) exp(2 pi i k dot R). `--ws-mode apply` first distributes each orbital-pair hopping across its matching WS-vector shifts with the appropriate averaging weights. The resulting ordinary HR contains that interpolation correction. Never divide by source degeneracies again or reapply the original WS vectors to the delivered file.

Choose the mode from the model's intended interpolation and actual Wannier90 settings, not whichever produces a smoother plot. Where native Wannier90 bands exist, compare the original HR interpolation to them before interpreting symmetrization changes. The collinear and magnetic runners offer optional `native_bands` checks; perform an equivalent explicit check for the nonmagnetic runner when needed.

Every completed branch requires `band_path` and produces `bands.npz` plus `bands_before_after.png`, `.pdf`, `.svg`. Use matching k points, path breaks, reciprocal metric, one energy reference and the same interpolation convention for before/after. ISPIN=2 requires separate channel comparisons; spinor cases compare the full spectrum. Report maximum and RMS eigenvalue changes and inspect the overlay. The comparison pairs sorted energies at each k, not tracked eigenvectors across crossings. Generic-k and TRIM tests supplement this final path comparison.

A small residual proves numerical covariance for the assumed representation; it cannot prove a correct gauge or DFT fit. Large band changes require diagnosis of axes, atomic gauge, interpolation, moment symmetry and original Wannier quality. When DFT bands are available they provide an additional physical check, without replacing the HR-before/after comparison.
