# Case description

The agent prepares a JSON object from the model's actual provenance. Shared fields are shown below for a synthetic scalar s orbital; do not copy its physical values into a real model.

```json
{
  "magnetic": false,
  "spinor": false,
  "spin_order": "scalar",
  "axes_convention": "physical-cartesian-columns",
  "lattice_A": [[4,0,0],[0,4,0],[0,0,4]],
  "atoms": [{"element":"X","fractional":[0,0,0]}],
  "sites": [{"atom":1,"shells":["s"],"x":[1,0,0],"z":[0,0,1]}],
  "evidence": {
    "magnetic_state":"Synthetic nonmagnetic model",
    "orbital_order":"One real s orbital",
    "local_axes":"Cartesian identity",
    "home_cells":"Origin",
    "spin_order":"Scalar",
    "versions":"Synthetic model; no DFT interface"
  },
  "band_path": [{"start":[0,0,0],"end":[0.5,0,0],"labels":["Gamma","X"],"points":51}]
}
```

- `lattice_A`: lattice vectors as rows, in Angstrom, including any POSCAR scale factor.
- `atoms`: all structural atoms, including those outside the projected basis. Fractional coordinates include verified integer home-cell shifts; keep ideal atomic sites, not intra-cell-displaced Wannier centers.
- `sites`: projected site blocks in actual orbital order. `atom` is one-based; each atom appears at most once. `shells` supports s, p, d, f and t2g, subject to closure under the selected symmetry. Combine shells belonging to one site in their actual order.
- `x` and `z`: physical Cartesian local axes, normalized internally and required to be orthogonal. The right-handed y axis is z cross x. These are orbital axes, not spin axes.
- `evidence`: nonempty provenance strings. Each branch adds specific required fields. Their presence is not independent proof of physical correctness.
- `fermi_eV`: optional common energy reference. Omit when unknown; plots then show absolute energy. Never independently align original/symmetrized or up/down spectra.
- `plot_window_eV`: optional display limits relative to the common Fermi reference, or absolute energies when none is given. Does not restrict numerical comparisons.
- `band_path`: required for completed symmetrization. Each segment supplies fractional reciprocal start/end vectors, two labels and `points >= 2`. Segment breaks are retained.

Nonmagnetic SOC uses `magnetic=false`, `spinor=true`, `spin_order="all-up-then-all-down"`. The spatial `sites` array is not duplicated for spin. Other branch-specific fields are documented in [collinear](collinear.md) and [magnetic](magnetic.md) and [SSG](ssg.md).

Outputs use degeneracy weights one. Depending on branch, they include original/prepared HR copies, the delivered HR, case and manifest, operation representations, validation metrics, generic-k spectra, path spectra and PNG/PDF/SVG overlays. Consult the entrypoint's `--help` for exact CLI arguments; existing output directories are refused.
