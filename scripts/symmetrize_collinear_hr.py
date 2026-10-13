#!/usr/bin/env python3
"""Joint Python 3 symmetrization of collinear, non-SOC up/down Wannier HRs."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import numpy as np

CORE = Path(__file__).resolve().parent / 'wannsym_core'
sys.path.insert(0, str(CORE))
import wannsym as sw
import collinear as cl
import symmetrize_hr as common

VERSION = '1.2.0'
check = common.check
digest = common.digest


def basis_from_config(cfg):
    check(cfg.get('magnetic') is True and cfg.get('spinor') is False,
          'Requires a confirmed collinear magnetic, non-SOC scalar-channel model')
    check(cfg.get('spin_order') == 'separate-up-down', 'Expected separate-up-down order')
    check(cfg.get('axes_convention') == 'physical-cartesian-columns', 'Unaudited physical axes')
    check(type(cfg.get('real_gauge')) is bool, 'real_gauge must be an explicit boolean')
    evidence = cfg.get('evidence', {})
    for key in ['magnetic_state', 'orbital_order', 'local_axes', 'home_cells', 'spin_order',
                'versions', 'common_gauge', 'real_gauge']:
        check(isinstance(evidence.get(key), str) and evidence[key].strip(), 'Missing evidence: '+key)
    lattice = np.asarray(cfg['lattice_A'], float).T
    check(lattice.shape == (3,3) and np.isfinite(lattice).all() and abs(np.linalg.det(lattice)) > 1e-12,
          'Invalid lattice vectors')
    atoms, sites = cfg['atoms'], cfg['sites']
    pos = np.asarray([a['fractional'] for a in atoms], float)
    symbols = [a['element'] for a in atoms]
    check(len(atoms) > 0 and pos.shape == (len(atoms),3) and np.isfinite(pos).all(), 'Invalid atoms')
    check(all(isinstance(s,str) and s and len(s.split()) == 1 for s in symbols), 'Invalid species')
    ids, shells, axes, sizes = [], [], [], []
    for site in sites:
        i = site['atom']
        check(type(i) is int and 1 <= i <= len(atoms) and i-1 not in ids, 'Invalid/duplicate projected atom')
        orbs = site['shells']
        check(isinstance(orbs,list) and orbs and all(o in sw.DIMS for o in orbs), 'Invalid shells')
        x, z = np.asarray(site['x'],float), np.asarray(site['z'],float)
        check(x.shape == z.shape == (3,) and np.isfinite([x,z]).all(), 'Invalid local axes')
        check(min(np.linalg.norm(x),np.linalg.norm(z)) > 1e-12, 'Zero local axis')
        x, z = x/np.linalg.norm(x), z/np.linalg.norm(z)
        check(abs(x@z) < 1e-7, 'Nonorthogonal local axes')
        ids.append(i-1); shells.append(orbs); axes.append(np.column_stack((x,np.cross(z,x),z)))
        sizes.append(sum(sw.DIMS[o] for o in orbs))
    check(bool(ids), 'Empty orbital basis')
    offsets = np.cumsum([0]+sizes)
    slices = [slice(int(a),int(b)) for a,b in zip(offsets[:-1],offsets[1:])]
    return sw.Basis(lattice,pos,symbols,ids,shells,np.asarray(axes),False,sizes,slices,int(offsets[-1]))


def pair_difference(a,b,n):
    zero = np.zeros((n,n),complex)
    return max(float(np.max(abs(x.get(r,zero)-y.get(r,zero))))
               for x,y in zip(a,b) for r in set(x)|set(y))


def native_check(h, spec):
    k = np.loadtxt(spec['kpt'],skiprows=1)[:,:3]
    rows = np.loadtxt(spec['dat'])
    n = next(iter(h.values())).shape[0]
    check(len(rows) == len(k)*n, 'Native Wannier90 band size mismatch')
    ref = np.sort(rows[:,1].reshape(n,len(k)).T,axis=1)
    energy, herm = common.spectrum(h,k)
    err = float(abs(energy-ref).max())
    check(err <= spec.get('tolerance_eV',1e-4), f'Native Wannier90 band mismatch: {err} eV')
    return {'max_error_eV':err, 'tolerance_eV':spec.get('tolerance_eV',1e-4),
            'points':len(k),'Bloch_hermiticity_eV':herm,
            'sources':{str(Path(spec[s]).resolve()):digest(spec[s]) for s in ['kpt','dat']}}


def band_summary(before,after,ef):
    delta = after-before
    ans = {'max_change_eV':float(abs(delta).max()), 'RMS_change_eV':float(np.sqrt(np.mean(delta**2)))}
    if ef is not None:
        mask = (abs(before-ef) <= 1) | (abs(after-ef) <= 1)
        ans['within_1eV_of_EF'] = {'count':int(mask.sum()),
            'max_change_eV':float(abs(delta[mask]).max()) if mask.any() else None,
            'RMS_change_eV':float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None}
    return ans


def plot(out,x,parts,ticks,labels,before,after,ef,window):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.lines import Line2D
    from plot_style import configure_plot_style
    font = configure_plot_style()
    fig, ax = plt.subplots(2,2,figsize=(20,10),sharex='col',layout='constrained',
                           gridspec_kw={'height_ratios':[3,1]})
    zero = 0 if ef is None else ef
    scale = 1e6 if abs(after-before).max() < 1e-3 else 1.
    delta_unit = 'μeV' if scale == 1e6 else 'eV'
    for s,(name,color) in enumerate([('Spin up','#d62728'),('Spin down','#1f77b4')]):
        for a,b in parts:
            ax[0,s].plot(x[a:b],before[s,a:b]-zero,color='0.35',lw=1.25,alpha=.75)
            ax[0,s].plot(x[a:b],after[s,a:b]-zero,color=color,lw=1.35,ls='--')
            ax[1,s].plot(x[a:b],scale*abs(after[s,a:b]-before[s,a:b]).max(axis=1),color=color,lw=1.5)
        ax[0,s].set_title(name)
        if window is not None: ax[0,s].set_ylim(*window)
        if ef is not None: ax[0,s].axhline(0,color='black',lw=1,ls=':')
        ax[0,s].legend(handles=[Line2D([],[],color='0.35',label='Before'),
                                Line2D([],[],color=color,ls='--',label='After')],fontsize=22,ncol=2)
        ax[1,s].set_xticks(ticks,[v.replace('Gamma','Γ') for v in labels])
        ax[1,s].set_ylim(bottom=0); ax[1,s].set_xlabel('Wave-vector path')
        for axis in ax[:,s]:
            axis.set_xlim(x[0],x[-1]); axis.tick_params(direction='in',labelsize=30)
            for t in ticks: axis.axvline(t,color='black',lw=.6)
    ax[0,0].set_ylabel('E - E_F (eV)' if ef is not None else 'Energy (eV)')
    ax[1,0].set_ylabel('max |delta E|\n('+delta_unit+')',fontsize=22)
    for ext in ['png','pdf','svg']: fig.savefig(out/f'bands_before_after.{ext}',dpi=220)
    plt.close(fig)
    return font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--up-hr',type=Path,required=True)
    parser.add_argument('--down-hr',type=Path,required=True)
    parser.add_argument('--case-json',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ws-mode',choices=['raw','apply'],required=True)
    parser.add_argument('--up-wsvec',type=Path)
    parser.add_argument('--down-wsvec',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--symprec',type=float,default=1e-5)
    parser.add_argument('--moment-tol',type=float,default=1e-3)
    parser.add_argument('--validation-tol',type=float,default=1e-8)
    parser.add_argument('--exchange-operation',type=int,
                        help='Zero-based index in operations.json; default: first channel-exchanging operation')
    args = parser.parse_args()
    check(min(args.symprec,args.moment_tol,args.validation_tol) > 0, 'Invalid tolerance')
    cfg = json.loads(args.case_json.read_text())
    check(args.prepare_only or cfg.get('band_path'), 'A verified before/after band_path is required')
    basis = basis_from_config(cfg)
    ops = cl.select_operations(basis,cfg['moments_muB'],args.symprec,args.moment_tol)
    projection_info = cl.projection_metadata(ops,exchange_index=args.exchange_operation)
    out = args.out.resolve()
    check(not out.exists(), 'Output directory exists; choose a fresh directory')
    sources = [args.up_hr.resolve(),args.down_hr.resolve()]
    check(sources[0] != sources[1], 'Up and down must be independently identified source files')
    ws = [args.up_wsvec,args.down_wsvec]
    check(all(p is not None for p in ws) if args.ws_mode == 'apply' else all(p is None for p in ws),
          'Apply requires both WS files; raw requires neither')
    pair = [sw.read_hr(p) for p in sources]
    check(all(next(iter(h.values())).shape == (basis.n,basis.n) for h in pair),'HR/basis dimensions disagree')
    out.mkdir(parents=True)
    (out/'case.json').write_text(json.dumps(cfg,indent=2)+'\n')
    manifest = {'runner_version':VERSION,'core_version':cl.VERSION,'python':platform.python_version(),
        'numpy':np.__version__,'spglib':sw.spglib.__version__,'sympy':sw.sympy.__version__,
        'runner_sha256':digest(__file__),
        'numerical_sources':{str(p):digest(p) for p in
            [CORE/'wannsym.py',CORE/'collinear.py',Path(common.__file__)]+sorted((CORE/'lib').glob('*.py'))},
        'config_sha256':digest(out/'case.json'),'sources':[], 'ws_mode':args.ws_mode,
        'output_interpolation':'Ordinary weighted HR Fourier sum; output weights=1; do not reapply WS vectors',
        'projection':projection_info,
        'status':'prepared'}
    for s,name in enumerate(['up','down']):
        sh = digest(sources[s]); shutil.copy2(sources[s],out/f'original_{name}_hr.dat')
        check(digest(out/f'original_{name}_hr.dat') == sh,'HR changed during preparation')
        pair[s] = sw.read_hr(out/f'original_{name}_hr.dat')
        check(next(iter(pair[s].values())).shape == (basis.n,basis.n),'Snapshot HR/basis dimensions disagree')
        rec = {'channel':name,'hr':str(sources[s]),'sha256':sh}
        if args.ws_mode == 'apply':
            wh = digest(ws[s])
            shutil.copy2(ws[s],out/f'source_{name}_wsvec.dat')
            check(digest(out/f'source_{name}_wsvec.dat') == wh,'WS source changed during preparation')
            pair[s] = common.apply_ws(pair[s],out/f'source_{name}_wsvec.dat')
            rec.update(wsvec=str(ws[s].resolve()),wsvec_sha256=wh)
        manifest['sources'].append(rec)
        sw.write_hr(out/f'prepared_{name}_hr.dat',pair[s],'Prepared weighted scalar channel; weights one')
    (out/'operations.json').write_text(json.dumps(cl.operation_metadata(ops),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.prepare_only:
        print('PREPARED',out); return
    summary = {'projection':projection_info,
               'operation_counts':{'preserving':sum(o.eta==1 for o in ops),'exchanging':sum(o.eta==-1 for o in ops)},
               'before_residuals':cl.residuals(pair,basis,ops),'input_R_counts':[len(h) for h in pair]}
    if cfg.get('native_bands'):
        check(args.ws_mode == 'apply','Native band verification requires audited WS interpolation')
        check(len(cfg['native_bands']) == 2,'Native band verification requires both channels')
        summary['native_wannier90_checks'] = [native_check(h,s) for h,s in zip(pair,cfg['native_bands'])]
    result = cl.joint_project(pair,basis,ops,real_gauge=cfg['real_gauge'],exchange_index=args.exchange_operation)
    for h,name in zip(result,['up','down']):
        sw.write_hr(out/f'wannier90_{name}_hr_sym.dat',h,'Collinear non-SOC joint symmetry projection; weights one')
    # Read the delivered text files, then verify the actual output.
    result = tuple(sw.read_hr(out/f'wannier90_{name}_hr_sym.dat') for name in ['up','down'])
    summary['output_R_counts'] = [len(h) for h in result]
    summary['after_residuals'] = cl.residuals(result,basis,ops)
    for key in ['magnetic_group_max_eV','hermiticity_max_eV'] + (['real_gauge_K_max_eV'] if cfg['real_gauge'] else []):
        check(summary['after_residuals'][key] <= args.validation_tol,
              f'Output residual failed: {key}={summary["after_residuals"][key]}')
    repeated = cl.joint_project(result,basis,ops,real_gauge=cfg['real_gauge'],exchange_index=args.exchange_operation)
    repeat = pair_difference(result,repeated,basis.n)
    summary['repeat_projection_max_eV'] = repeat
    check(repeat <= args.validation_tol,f'Repeat projection failed: {repeat}')
    k = np.random.default_rng(2718).random((16,3))
    before_data = [common.spectrum(h,k) for h in pair]
    after_data = [common.spectrum(h,k) for h in result]
    before = np.stack([v[0] for v in before_data])
    after = np.stack([v[0] for v in after_data])
    summary['Bloch_hermiticity_before_eV'] = [v[1] for v in before_data]
    summary['Bloch_hermiticity_after_eV'] = [v[1] for v in after_data]
    summary['band_diagonalization'] = 'Eigenvalues of (H(k)+H(k)^dagger)/2; unmodified Bloch Hermiticity residual reported'
    check(max(summary['Bloch_hermiticity_after_eV']) <= args.validation_tol*max(len(h) for h in result),
          'Output Bloch Hamiltonian is not Hermitian')
    np.savez_compressed(out/'generic_k_bands.npz',kpoints=k,before_eV=before,after_eV=after)
    summary['generic_bands'] = [band_summary(b,a,cfg.get('fermi_eV')) for b,a in zip(before,after)]
    summary['same_k_spin_splitting_max_before_after_eV'] = [float(abs(e[0]-e[1]).max()) for e in [before,after]]
    cov = 0.
    for tagged in ops:
        kg = k[:4] @ np.linalg.inv(tagged.op.rotation)
        for s in [0,1]:
            target = s if tagged.eta == 1 else 1-s
            e,_ = common.spectrum(result[target],kg)
            cov = max(cov,float(abs(e-after[s,:4]).max()))
    summary['generic_k_spin_group_spectrum_max_eV'] = cov
    check(cov <= args.validation_tol*max(len(h) for h in result),f'Spectral covariance failed: {cov}')
    if cfg.get('band_path'):
        k,x,parts,ticks,labels = common.band_path(cfg)
        before = np.stack([common.spectrum(h,k)[0] for h in pair])
        after = np.stack([common.spectrum(h,k)[0] for h in result])
        np.savez_compressed(out/'bands.npz',kpoints=k,distance=x,segments=parts,before_eV=before,after_eV=after,delta_eV=after-before)
        summary['path_bands'] = [band_summary(b,a,cfg.get('fermi_eV')) for b,a in zip(before,after)]
        summary['energy_pairing'] = 'Sorted eigenvalues at each k in each spin channel; no eigenvector tracking'
        summary['plot_font'] = plot(out,x,parts,ticks,labels,before,after,cfg.get('fermi_eV'),cfg.get('plot_window_eV'))
    for rec in manifest['sources']: check(digest(rec['hr']) == rec['sha256'],'Source HR changed during execution')
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest['status'] = 'completed'
    manifest['outputs'] = {name:digest(out/name) for name in ['wannier90_up_hr_sym.dat','wannier90_down_hr_sym.dat']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('COMPLETED',out,json.dumps(summary,indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError,OSError,KeyError,IndexError,TypeError) as error:
        print('ERROR:',error,file=sys.stderr)
        raise SystemExit(2)
