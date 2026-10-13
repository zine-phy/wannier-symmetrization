# Setup, execution and tests

Use Python 3.10 or newer with the dependencies in `requirements.txt`. Install into an existing suitable environment or a dedicated environment; no particular environment manager is required. Paths in the commands below are relative to the skill root.

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

The numerical tests cover scalar and spinor nonmagnetic projection, paired collinear projection, semilinear MSG projection and noncollinear SSG projection. Portable CLI tests copy the entire skill into a temporary directory with spaces, run from a different directory with a clean import path, and exercise all five physical branches including plots. No server fixtures or network access are needed for these tests.

Generate small runnable examples with `python3 examples/make_demo.py --out demo`. The output includes five case descriptions and their input HRs, plus POSCAR/WIN/WOUT fixture subdirectories for the automatic interface; these are synthetic algebraic examples, not material templates. See `examples/make_demo.py --help` and the branch references for commands.

| Component | Location |
| --- | --- |
| Automatic POSCAR interface | `scripts/symmetrize.py` and `scripts/auto_inputs.py` |
| Nonmagnetic runner | `scripts/symmetrize_hr.py` |
| Paired collinear runner | `scripts/symmetrize_collinear_hr.py` |
| Magnetic spinor MSG runner | `scripts/symmetrize_magnetic_hr.py` |
| Noncollinear NSOC SSG runner | `scripts/symmetrize_ssg_hr.py` |
| Numerical cores and orbital/spin rotations | `scripts/wannsym_core/` |
| Shared plotting configuration | `scripts/plot_style.py` |
| Analytic and relocation tests | `tests/` |

Entrypoints locate bundled modules from their own file paths. They do not require `PYTHONPATH`, a current working directory equal to the skill root, or a separate core installation. CLI file arguments are resolved relative to the process working directory. Paths inside optional `native_bands` entries follow that same convention; use known resolved paths when preparing a case for another working directory.

Set BLAS/OpenMP thread counts to allocated CPUs on shared systems. Choose memory and wall time from HR dimension/R support and follow local scheduler instructions. Plotting uses Times New Roman when installed, otherwise the Matplotlib-bundled DejaVu Serif. `WANNSYM_FONT` can explicitly select another installed font; an unavailable explicit selection fails rather than silently substituting. The chosen font is recorded in the result.

Only source, documentation, tests and synthetic example generators belong in the distributable folder. Keep calculation outputs, credentials, site settings and private input files outside it. No GitHub publication is performed by this skill.
