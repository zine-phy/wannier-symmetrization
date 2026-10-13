---
name: wannsym
description: Classify and symmetrize Wannier HR models with portable Python 3 tools, audited orbital and spin bases, and mandatory before/after band comparisons. Supports nonmagnetic scalar/SOC and collinear non-SOC up/down models, plus experimental magnetic spinor MSG and noncollinear non-SOC SSG workflows.
---

# Wannier HR Symmetrization

All code needed by this skill is bundled in this directory. Resolve scripts and references relative to this file, not the current working directory or a remembered installation. Use the active environment's Python 3 and local resource/scheduler instructions. No remote account, fixed Conda environment, private data directory, or separately installed Wannsym core is required. See [setup and tests](references/setup.md).

## 1. Classify the physical model

Inspect matching structure, projections, Wannier output and magnetic/SOC settings before choosing a runner. NSOC means no spin-orbit coupling. Zero total magnetization does not establish nonmagnetism; `ISPIN` alone does not classify a SOC/noncollinear model. Record the selected branch and evidence.

| Branch | Actual representation | Workflow |
| --- | --- | --- |
| Nonmagnetic SOC | TR-invariant state; complete spinor HR | Implemented: [nonmagnetic](references/nonmagnetic.md), `scripts/symmetrize_hr.py` |
| Nonmagnetic NSOC | Nonmagnetic scalar orbital HR | Implemented: [nonmagnetic](references/nonmagnetic.md), `scripts/symmetrize_hr.py` |
| Magnetic collinear NSOC, ISPIN=2 | Separate scalar up/down HRs; SOC and noncollinearity disabled | Implemented for AFM, FM and ferrimagnetic moment patterns: [collinear](references/collinear.md), `scripts/symmetrize_collinear_hr.py` |
| Magnetic SOC | Magnetic complete spinor HR, whether the moments are collinear or noncollinear | Experimental MSG implementation: [magnetic](references/magnetic.md), `scripts/symmetrize_magnetic_hr.py` |
| Magnetic noncollinear NSOC | Complete spinor HR without SOC | Experimental SSG implementation for coplanar and noncoplanar textures: [SSG](references/ssg.md), `scripts/symmetrize_ssg_hr.py`; includes the coplanar spin-only mirror. |

A nonmagnetic spinor calculation with SOC disabled is not automatically a scalar input: establish a valid reduction first. An individual magnetic up-channel file is not a nonmagnetic model. Do not apply the nonmagnetic runner or pure spin time reversal to an ordered magnet.

## 2. Audit and prepare

An HR alone does not specify its physical basis. Recover matching orbital order, spin order/frame, local orbital axes, atomic home cells, magnetic moments, energy zero and interpolation convention. Read [basis and source conventions](references/basis-and-axes.md), and the selected branch's input schema. Record actual VASP/Wannier90 versions or explicitly mark them unknown.

Use the [automatic POSCAR interface](references/automatic-inputs.md), `scripts/symmetrize.py`, to generate `case.json` and core inputs from POSCAR moments, calculation type, HR and matching WIN/WOUT metadata. Do not require the user to construct internal JSON manually. Ask only for essential information that cannot be recovered. Preserve original inputs and use a fresh output directory. Arbitrary MLWF gauges require a known transformation; local-axis relabeling is not gauge alignment. Keep symmetrization before subsequent orbital reordering unless the transformation is explicitly included.

The unified entrypoint offers `--ws-mode auto` and records how it resolves metadata. For an explicit choice, use `--ws-mode apply` with matching WS files when reproducing a model that uses WS-distance interpolation, or `--ws-mode raw` for an explicitly intended bare-HR convention. Never silently switch conventions or reapply source WS vectors to the delivered HR. Read [interpolation and bands](references/interpolation-and-bands.md).

## 3. Run and validate

Run the matching bundled entrypoint using the existing compute environment. Follow the environment's scheduler rules for substantial calculations; this skill does not prescribe a machine, partition or account. Inspection/preparation-only requests stop at the requested stage.

Check successful exit, completed manifest, delivered-file symmetry and Hermiticity, repeat projection, and branch-specific checks. Numerical symmetry alone does not certify the input basis or preserve a DFT fit. Validation scope is summarized in [validation](references/validation.md).

## 4. Always finish with before/after bands

Every completed symmetrization ends with an inspected band overlay and numerical maximum/RMS changes on the same verified k path, sampling, energy reference and interpolation convention. The unified entrypoint recovers WIN `kpoint_path` or generates a SeeK-path comparison path in the original reciprocal basis. Low-level runners require `band_path`; completed execution without it is rejected. Preparation-only modes may omit it.

Compare each ISPIN=2 channel separately; compare the complete spinor spectrum for SOC/noncollinear HRs. Generic-k checks and a DFT overlay do not replace this final step. Recover the path from the project or establish a crystallographically justified one. If essential information remains unresolved, report the workflow as incomplete.

Return the final HR location, branch, source/code versions, interpolation convention, residuals, band changes and PNG/PDF/SVG plots. Explain substantial changes before accepting the model. Runtime manifests may record the user's actual paths for provenance; those generated files and private data are not part of the distributable skill.
