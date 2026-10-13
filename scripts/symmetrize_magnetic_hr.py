#!/usr/bin/env python3
"""Project a spinor HR onto an audited magnetic space group, then compare bands."""
import argparse,json,os,sys,platform,shutil
from pathlib import Path
import numpy as np
import scipy,spglib
CORE=Path(__file__).resolve().parent / 'wannsym_core'
sys.path.insert(0,str(CORE))
import wannsym as w
import magnetic as msg
import symmetrize_hr as common
from symmetrize_collinear_hr import native_check

VERSION='0.3.0'

def basis_from_config(cfg, target="MSG"):
    w.require(cfg.get('magnetic') is True and cfg.get('spinor') is True,'Requires magnetic full spinor model')
    w.require(target in ('MSG','SSG') and cfg.get('target_symmetry')==target, 'Target symmetry does not match runner')
    if target=='SSG': w.require(cfg.get('source_soc') is False, 'SSG independent spin rotations require source_soc=false')
    w.require(type(cfg.get('source_soc')) is bool,'Declare actual source_soc')
    w.require(cfg.get('spin_order') in ['interleaved','all-up-then-all-down'],'Unsupported spin order')
    w.require(cfg.get('spin_frame')=='global-cartesian','Convert spin basis and moments to global Cartesian frame first')
    w.require(cfg.get('axes_convention')=='physical-cartesian-columns','Unaudited axes')
    for key in ['magnetic_state','orbital_order','local_axes','home_cells','spin_order','spin_frame','versions','atomic_gauge','moments']:
        w.require(isinstance(cfg.get('evidence',{}).get(key),str) and cfg['evidence'][key].strip(),'Missing evidence: '+key)
    lattice=np.asarray(cfg['lattice_A'],float).T
    w.require(lattice.shape==(3,3) and np.isfinite(lattice).all() and abs(np.linalg.det(lattice))>1e-12,'Invalid lattice')
    atoms=cfg['atoms']; pos=np.asarray([a['fractional'] for a in atoms],float)
    w.require(pos.shape==(len(atoms),3) and np.isfinite(pos).all(),'Invalid positions')
    symbols=[a['element'] for a in atoms]
    ids=[]; axes=[]; shells=[]; sizes=[]
    for site in cfg['sites']:
        i=site['atom']; orbs=site['shells']
        w.require(type(i) is int and 1<=i<=len(atoms) and i-1 not in ids,'Invalid projected site')
        w.require(orbs and all(o in w.DIMS for o in orbs),'Unsupported shells')
        x,z=np.asarray(site['x'],float),np.asarray(site['z'],float)
        w.require(x.shape==z.shape==(3,) and np.isfinite([x,z]).all() and min(np.linalg.norm(x),np.linalg.norm(z))>1e-12,'Invalid axes')
        x=x/np.linalg.norm(x); z=z/np.linalg.norm(z)
        w.require(abs(x@z)<1e-7,'Nonorthogonal axes')
        ids.append(i-1); axes.append(np.column_stack((x,np.cross(z,x),z))); shells.append(orbs); sizes.append(2*sum(w.DIMS[o] for o in orbs))
    w.require(bool(ids),'Empty basis')
    offsets=np.cumsum([0]+sizes)
    return w.Basis(lattice,pos,symbols,ids,shells,np.asarray(axes),True,sizes,[slice(int(a),int(b)) for a,b in zip(offsets[:-1],offsets[1:])],int(offsets[-1]))


def internal(h,cfg,inverse=False):
    return w.reorder(h,True,inverse=inverse) if cfg['spin_order']=='all-up-then-all-down' else h


def band_metrics(a,b):
    d=b-a
    return {'max_change_eV':float(abs(d).max()),'RMS_change_eV':float(np.sqrt(np.mean(d*d)))}


def main(target="MSG"):
    if target=="SSG":
        import ssg as engine
    else:
        engine=msg
    p=argparse.ArgumentParser(description="Spinor HR projection onto "+target+" with final band comparison")
    p.add_argument('--hr',type=Path,required=True); p.add_argument('--case-json',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--ws-mode',choices=['raw','apply'],required=True); p.add_argument('--wsvec',type=Path)
    p.add_argument('--symprec',type=float,default=1e-5); p.add_argument('--moment-tol',type=float,default=1e-3); p.add_argument('--validation-tol',type=float,default=1e-7)
    if target=='SSG': p.add_argument('--rank-tol',type=float,default=1e-6)
    a=p.parse_args(); cfg=json.loads(a.case_json.read_text()); basis=basis_from_config(cfg,target)
    w.require(min(a.symprec,a.moment_tol,a.validation_tol)>0,'Invalid tolerances')
    w.require(cfg.get('band_path'),'A verified before/after band path is required')
    w.require((a.ws_mode=='apply')==(a.wsvec is not None),'Supply matching WS data exactly for apply mode')
    w.require(not a.out.exists(),'Output directory already exists')
    options={'rank_tol':a.rank_tol} if target=='SSG' else {}
    ops=engine.select_operations(basis,cfg['moments_cartesian_muB'],a.symprec,a.moment_tol,**options)
    a.out.mkdir(parents=True)
    sources={str(x.resolve()):common.digest(x) for x in [a.hr,a.case_json]+([a.wsvec] if a.wsvec else [])}
    manifest={'status':'running','runner_version':VERSION,'core_version':engine.VERSION,'sources':sources,
              'source_soc':cfg['source_soc'],'target':target,'scope':('Non-SOC noncollinear SSG, including the coplanar spin-only stabilizer' if target=='SSG' else ('SOC magnetic model' if cfg['source_soc'] else 'MSG-subgroup test on an NSOC spinor model; not full SSG')),
              'spin_order':cfg['spin_order'],'spin_frame':cfg['spin_frame'],'ws_mode':a.ws_mode,
              'operations':len(ops),'unitary':sum(not g.antiunitary for g in ops),'antiunitary':sum(g.antiunitary for g in ops),
              'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'spglib':spglib.__version__},
              'symprec_A':a.symprec,'moment_tolerance_muB':a.moment_tol,'validation_tolerance_eV':a.validation_tol,
              'code_hashes':{str(x):common.digest(x) for x in [Path(__file__).resolve(),Path(engine.__file__),Path(msg.__file__),Path(w.__file__),Path(common.__file__)]}}
    if target=='SSG': manifest.update(moment_rank=ops[0].moment_rank,rank_tolerance_muB=a.rank_tol)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copy2(a.hr,a.out/'original_hr.dat'); shutil.copy2(a.case_json,a.out/'case.json')
    if a.wsvec: shutil.copy2(a.wsvec,a.out/'source_wsvec.dat')
    (a.out/'operations.json').write_text(json.dumps(engine.metadata(ops),indent=2))
    h=w.read_hr(a.hr); raw_count=len(h)
    if a.wsvec: h=common.apply_ws(h,a.wsvec)
    native=native_check(h,cfg['native_bands']) if cfg.get('native_bands') else None
    h=internal(h,cfg); engine.validate_hr(h,basis)
    before_res=engine.residuals(h,basis,ops)
    w.write_hr(a.out/'prepared_hr.dat',internal(h,cfg,True),'Prepared magnetic spinor HR; weights one; WS already absorbed')
    final=engine.project(h,basis,ops)
    output=a.out/'wannier90_hr_sym.dat'
    w.write_hr(output,internal(final,cfg,True),'Python 3 '+target+' projection; weights one; no old WS reapplication')
    delivered=internal(w.read_hr(output),cfg)
    after_res=engine.residuals(delivered,basis,ops)
    repeat=engine.difference(delivered,engine.project(delivered,basis,ops),basis.n)
    ks=np.random.default_rng(20261012).uniform(-.5,.5,(16,3))
    bloch=engine.bloch_residual(delivered,basis,ops,ks[:5])
    e0,bh0=common.spectrum(h,ks); e1,bh1=common.spectrum(delivered,ks)
    path,x,parts,ticks,labels=common.band_path(cfg)
    b0,ph0=common.spectrum(h,path); b1,ph1=common.spectrum(delivered,path)
    val={'before':before_res,'after':after_res,'repeat_projection_max_eV':repeat,'Bloch_covariance_max_eV':bloch,
         'native_bands':native,'generic_bands':band_metrics(e0,e1),'path_bands':band_metrics(b0,b1),
         'Bloch_hermiticity_before_after_eV':[max(bh0,ph0),max(bh1,ph1)],'raw_R':raw_count,'prepared_R':len(h),'output_R':len(delivered),'dimension':basis.n}
    (a.out/'validation.json').write_text(json.dumps(val,indent=2))
    w.require(max(after_res['unitary_max_eV'],after_res['antiunitary_max_eV'],after_res['hermiticity_max_eV'],repeat,bloch)<=a.validation_tol,'Projection validation failed')
    np.savez_compressed(a.out/'generic_k_bands.npz',k=ks,before=e0,after=e1)
    np.savez_compressed(a.out/'bands.npz',k=path,x=x,parts=parts,ticks=ticks,labels=labels,before=b0,after=b1)
    font=common.plot_bands(a.out,x,parts,ticks,labels,b0,b1,cfg.get('fermi_eV'),cfg.get('plot_window_eV'))
    w.require(all(common.digest(name)==sha for name,sha in sources.items()),'Input source changed during execution')
    manifest.update(status='completed',output_sha256=common.digest(output),font=font,physical_acceptance='Requires review of gauge evidence and reported band changes; numerical completion alone does not certify the DFT fit')
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'output':str(output),'unitary':manifest['unitary'],'antiunitary':manifest['antiunitary'],'validation':val},indent=2),flush=True)

if __name__=='__main__': main()
