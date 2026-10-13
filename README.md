# Wannier symmetrization

A portable Python 3 toolkit and English-language Codex skill for symmetrizing Wannier tight-binding Hamiltonians. Supply a POSCAR with Cartesian magnetic moments, the calculation type, and the HR. Matching Wannier metadata supply the orbital basis; internal input files are generated automatically.

Supports nonmagnetic NSOC/SOC, collinear magnetic NSOC with separate up/down HRs, magnetic SOC via magnetic space groups (MSG), and noncollinear magnetic NSOC via spin space groups (SSG). The magnetic spinor workflows are experimental; numerical symmetry does not establish a correct input gauge or preserve a DFT fit.

## Install and try

Python 3.10 or newer is required. From a checkout:

```bash
python3 -m pip install -r requirements.txt
python3 examples/make_demo.py --out demo
python3 scripts/symmetrize.py --poscar demo/scalar/POSCAR \
  --hr demo/scalar/wannier90_hr.dat \
  --magnetism nonmagnetic --ordering collinear --soc no --out runs/scalar
```

The demo includes synthetic inputs for all five branches. For noncollinear magnetic NSOC:

```bash
python3 scripts/symmetrize.py --poscar demo/ssg/POSCAR \
  --hr demo/ssg/wannier90_hr.dat \
  --magnetism magnetic --ordering noncollinear --soc no --out runs/ssg
```

For ISPIN=2:

```bash
python3 scripts/symmetrize.py --poscar demo/collinear/POSCAR \
  --up-hr demo/collinear/wannier90.up_hr.dat \
  --down-hr demo/collinear/wannier90.dn_hr.dat \
  --magnetism magnetic --ordering collinear --soc no --out runs/collinear
```

Append moments to POSCAR atom rows, after any Selective dynamics flags:

```text
Direct
0.5 0.5 0.5  1 2 3
0.0 0.0 0.0
```

Positions follow the POSCAR coordinate mode. Moments are always Cartesian muB; an omitted triple is zero. HR alone cannot identify orbital order or local axes. Matching WIN/WOUT metadata are read automatically; missing or ambiguous basis information requires an explicit assertion rather than guessing. See the [automatic interface](references/automatic-inputs.md) for all options and limitations.

Each completed run saves symmetry operations, symmetrized HR, numerical validation, and before/after bands in PNG/PDF/SVG and NPZ formats. Original inputs remain unchanged.

## Use as a skill

Clone this repository into a folder named `wannsym` within your tool's skill collection, or copy the complete repository there. `SKILL.md` is the entry point; all scripts and references are bundled. For Codex, for example:

```bash
git clone https://github.com/zine-phy/wannier-symmetrization.git "${CODEX_HOME:-$HOME/.codex}/skills/wannsym"
```

Use a fresh destination; preserve an existing installation before updating it. Dependencies must be installed in the Python environment used to run the scripts. There is no machine-specific account, absolute installation path, proprietary code, private calculation dataset, or MOM2SSG classification database dependency.

## Validation and attribution

```bash
python3 -m unittest discover -s tests -v
```

Tests exercise numerical invariants, all five workflows, relocation, automatic input construction and band artifacts. See [validation scope](references/validation.md) for the distinction between algorithm checks and physical validation. The optional [GitHub Actions template](ci/github-actions.yml) runs the same suite on Linux, macOS and Windows. To enable it, copy it to `.github/workflows/tests.yml` using GitHub credentials authorized to create workflows. Automated CI is not enabled by this initial publication.

Start with [SKILL.md](SKILL.md), [setup](references/setup.md), or the [input schema](references/input-schema.md) for advanced usage. This is an independent implementation drawing on WannierTools orbital rotation helpers and MOM2SSG operation-search logic, not an official WannierTools release. [Provenance and citations](references/provenance.md), [GPL-3.0 license](LICENSE).
