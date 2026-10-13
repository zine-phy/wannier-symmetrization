#!/usr/bin/env python3
"""Unified POSCAR + Cartesian moments + HR symmetrization interface."""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
import auto_inputs as ai
from symmetrize_hr import digest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'wannsym_core'))
import wannsym as core

VERSION = '1.0.0'


def header_dimension(path):
    with Path(path).open() as f:
        f.readline(); n = int(f.readline())
    ai.require(n > 0, 'Invalid HR dimension')
    return n


def locate(directory, explicit, preferred, suffix):
    if explicit:
        path = Path(explicit).resolve(); ai.require(path.is_file(), 'Missing metadata file: '+str(path)); return path
    for name in preferred:
        path = directory/name
        if path.is_file(): return path
    matches = list(directory.glob('*'+suffix)) if suffix else []
    ai.require(len(matches) <= 1, 'Multiple metadata files found; select explicitly: '+', '.join(str(p.name) for p in matches))
    return matches[0] if matches else None


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--poscar', type=Path, required=True)
    p.add_argument('--magnetism', choices=['magnetic', 'nonmagnetic'], required=True)
    p.add_argument('--ordering', choices=['collinear', 'noncollinear'], required=True)
    p.add_argument('--soc', choices=['yes', 'no'], required=True)
    p.add_argument('--hr', type=Path); p.add_argument('--up-hr', type=Path); p.add_argument('--down-hr', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--win', type=Path); p.add_argument('--wout', type=Path); p.add_argument('--outcar', type=Path)
    p.add_argument('--basis-json', type=Path, help='Audited existing case/basis object; sites and spin/home-cell metadata only')
    p.add_argument('--orbitals', help='Explicit species/site order, e.g. Mn:d;Sn:p or 1:d;2:p (Cartesian orbital axes)')
    p.add_argument('--spin-order', choices=['auto', 'interleaved', 'all-up-then-all-down'], default='auto')
    p.add_argument('--axes-convention', choices=['auto', 'physical', 'transpose'], default='auto')
    p.add_argument('--home-cells', choices=['auto', 'poscar'], default='auto')
    p.add_argument('--atomic-gauge', action='store_true', help='Confirm the model has the stated atomic real-harmonic gauge when num_iter=0 cannot establish provenance')
    p.add_argument('--spin-axis', nargs=3, type=float, help='Global Cartesian axis defining up/down for collinear channels; defaults to source SAXIS or +z')
    p.add_argument('--ws-mode', choices=['auto', 'raw', 'apply'], default='auto')
    p.add_argument('--wsvec', type=Path); p.add_argument('--up-wsvec', type=Path); p.add_argument('--down-wsvec', type=Path)
    p.add_argument('--fermi', type=float, help='Common energy zero in eV; no independent alignment')
    p.add_argument('--points', type=int, default=51, help='Samples per band-path segment')
    p.add_argument('--symprec', type=float, default=1e-5)
    p.add_argument('--moment-tol', type=float, default=1e-3)
    p.add_argument('--rank-tol', type=float, default=1e-6)
    p.add_argument('--prepare-only', action='store_true', help='Construct case and command without projecting HR')
    return p


def prepare(args):
    ai.require(min(args.symprec, args.moment_tol, args.rank_tol)>0 and args.points>=2, 'Invalid tolerance or sampling')
    magnetic = args.magnetism == 'magnetic'; soc = args.soc == 'yes'
    collinear = magnetic and not soc and args.ordering == 'collinear'
    ai.require(not(not magnetic and args.ordering == 'noncollinear'), 'Nonmagnetic input has no noncollinear magnetic order; use --ordering collinear')
    if collinear:
        ai.require(args.hr is None and args.up_hr and args.down_hr, 'Collinear magnetic NSOC requires --up-hr and --down-hr, not a single HR')
        hrs = [args.up_hr.resolve(), args.down_hr.resolve()]
        ai.require(hrs[0] != hrs[1] and hrs[0].parent == hrs[1].parent, 'Use distinct channels from one calculation directory')
    else:
        ai.require(args.hr and args.up_hr is None and args.down_hr is None, 'This branch requires one --hr')
        hrs = [args.hr.resolve()]
    ai.require(all(p.is_file() for p in hrs), 'Missing HR input')
    dimensions = [header_dimension(p) for p in hrs]
    ai.require(len(set(dimensions)) == 1, 'Up/down HR dimensions differ')
    dimension = dimensions[0]; directory = hrs[0].parent
    lattice, atoms, moments, explicit = ai.read_poscar(args.poscar)
    rank = int(np.sum(np.linalg.svd(moments, compute_uv=False) > args.rank_tol*np.sqrt(len(atoms))))
    if magnetic:
        ai.require(np.linalg.norm(moments, axis=1).max()>args.moment_tol, 'Magnetic case needs nonzero Cartesian moments appended to POSCAR')
        ai.require((args.ordering=='collinear' and rank==1) or (args.ordering=='noncollinear' and rank in (2,3)), 'Declared magnetic order conflicts with the supplied moment vectors')
    else:
        ai.require(np.max(abs(moments)) <= args.rank_tol, 'Nonmagnetic declaration conflicts with nonzero POSCAR moments')
    spinor = soc or (magnetic and not collinear)
    seed = re.sub(r'_hr\.dat.*$', '', hrs[0].name)
    win = locate(directory, args.win, [seed+'.win', 'wannier90.win'], '.win')
    wout = locate(directory, args.wout, [seed+'.wout', 'wannier90.wout'], '.wout')
    outcar = locate(directory, args.outcar, ['OUTCAR', 'OUTCAR.scf'], '')
    settings, blocks = ai.read_win(win); ai.check_win_structure(blocks, lattice)
    if 'spinors' in settings: ai.require(ai.truth(settings['spinors']) == spinor, 'WIN spinors disagrees with requested calculation type')
    if 'num_wann' in settings: ai.require(int(settings['num_wann']) == dimension, 'WIN num_wann differs from HR')
    source_versions = ['VASP version unknown (no matching OUTCAR)'] if outcar is None else []; source_axis = np.array([0.,0.,1.]); axis_known = False
    if outcar:
        with outcar.open(errors='replace') as f: text = f.read(2_000_000)
        version = re.search(r'vasp\.[^\n]+', text)
        source_versions.append(version[0].strip() if version else 'VASP version unknown')
        ispin = re.findall(r'\bISPIN\s*=\s*(\d+)',text)
        if ispin and not spinor: ai.require(int(ispin[-1])==(2 if collinear else 1),'OUTCAR ISPIN conflicts with scalar channel classification')
        for key, requested in [('LSORBIT', soc), ('LNONCOLLINEAR', spinor)]:
            found = re.findall(r'\b'+key+r'\s*=\s*(\.?[TF](?:RUE|ALSE)?\.?)', text, re.I)
            if found: ai.require(ai.truth(found[-1]) == requested, 'OUTCAR '+key+' conflicts with requested type')
        axes = re.findall(r'\bSAXIS\s*=\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)', text)
        if axes: source_axis=np.array([float(x) for x in axes[-1]]); axis_known=True
    original_atoms = copy.deepcopy(atoms)
    secondary_sources = []
    supplied_basis = json.loads(args.basis_json.read_text()) if args.basis_json else None
    ai.require(not(args.basis_json and args.orbitals), 'Choose either --basis-json or --orbitals')
    home_note = ''; release = 'unknown'
    if supplied_basis:
        if 'lattice_A' in supplied_basis: ai.require(np.allclose(supplied_basis['lattice_A'],lattice,atol=1e-6,rtol=1e-6),'Basis/POSCAR lattice mismatch')
        if 'spinor' in supplied_basis: ai.require(bool(supplied_basis['spinor'])==spinor,'Basis spinor representation differs from requested branch')
        sites = copy.deepcopy(supplied_basis['sites'])
        order = supplied_basis.get('spin_order', 'scalar' if not spinor else args.spin_order)
        ai.require(not spinor or order in ('interleaved','all-up-then-all-down'), 'Basis metadata must specify spin ordering')
        if 'atoms' in supplied_basis:
            old = supplied_basis['atoms']; ai.require(len(old)==len(atoms), 'Basis/POSCAR atom count mismatch')
            for a,b in zip(atoms,old):
                delta=np.array(b['fractional'])-np.array(a['fractional'])
                ai.require(a['element']==b['element'] and np.allclose(delta,np.rint(delta),atol=1e-6), 'Basis/POSCAR site mismatch')
                a['fractional']=b['fractional']
        ai.require(supplied_basis.get('axes_convention')=='physical-cartesian-columns', 'Basis JSON must declare physical Cartesian axes')
        home_note = 'Audited basis JSON home cells, or POSCAR representatives if absent'
    elif wout and not args.orbitals:
        sites,order,release,home_note = ai.basis_from_wout(wout,atoms,lattice,dimension,spinor,args.spin_order,args.axes_convention,args.home_cells)
    else:
        projected_blocks=copy.deepcopy(blocks)
        if args.orbitals:
            projected_blocks['projections']=[]
            for spec in args.orbitals.split(';'):
                selector,shells=spec.split(':',1)
                if selector.strip().isdigit():
                    index=int(selector)-1;ai.require(0<=index<len(atoms),'Invalid --orbitals site index')
                    selector='f='+','.join(map(str,atoms[index]['fractional']))
                projected_blocks['projections'].append(selector+':'+shells)
        sites,order=ai.basis_from_win(projected_blocks,atoms,spinor,args.spin_order,args.axes_convention)
        ai.require(args.home_cells=='poscar', 'Without final WOUT centers, confirm --home-cells poscar or supply audited --basis-json')
        home_note='Explicitly confirmed POSCAR home cells'
    if collinear and wout and not supplied_basis and not args.orbitals:
        down_seed = re.sub(r'_hr\.dat.*$', '', hrs[1].name)
        down_wout = directory/(down_seed+'.wout')
        if down_wout.is_file():
            down_atoms = copy.deepcopy(original_atoms)
            down_sites, _, _, _ = ai.basis_from_wout(down_wout, down_atoms, lattice, dimension, False, 'scalar', args.axes_convention, args.home_cells)
            ai.require(down_sites == sites and down_atoms == atoms, 'Up/down WOUT orbital bases or home cells differ; align the channel gauges first')
            secondary_sources.append(down_wout)
    expected=sum(sum(ai.DIMS[s] for s in site['shells']) for site in sites)*(2 if spinor else 1)
    ai.require(expected==dimension,'Constructed orbital basis does not match HR dimension')
    if not supplied_basis:
        ai.require(args.atomic_gauge or settings.get('num_iter','').strip()=='0', 'Cannot establish atomic gauge. Confirm --atomic-gauge after audit; arbitrary MLWF gauges cannot be inferred from POSCAR/HR')
    if spinor:
        ai.require(np.linalg.norm(source_axis)>0,'Invalid source spin axis')
        ai.require(np.allclose(source_axis/np.linalg.norm(source_axis),[0,0,1],atol=1e-7), 'Nondefault SAXIS requires conversion to global Cartesian spin frame before this runner')
        ai.require(not any(re.search(r'\b(?:spin|quant_dir)\s*=|\[[^]]+\]',line,re.I) for line in blocks.get('projections',[])), 'Custom projection spin axes require explicit spin-frame conversion')
    cfg={'magnetic':magnetic,'spinor':spinor,'spin_order':'separate-up-down' if collinear else order,
         'axes_convention':'physical-cartesian-columns','lattice_A':lattice.tolist(),'atoms':atoms,'sites':sites}
    if magnetic and spinor:
        cfg.update(target_symmetry='MSG' if soc else 'SSG',source_soc=soc,spin_frame='global-cartesian',moments_cartesian_muB=moments.tolist())
    if collinear:
        axis=np.array(args.spin_axis if args.spin_axis is not None else source_axis,float)
        ai.require(np.linalg.norm(axis)>0,'Zero collinear spin axis');axis/=np.linalg.norm(axis)
        signed=moments@axis
        ai.require(np.max(np.linalg.norm(moments-signed[:,None]*axis,axis=1))<=args.moment_tol,'Moments are not aligned with the up/down axis; specify --spin-axis')
        cfg.update(moments_muB=signed.tolist(),real_gauge=True)
    path,note=ai.path_from_metadata(blocks,lattice,atoms,magnetic,args.points,args.symprec)
    cfg['band_path']=path
    ef=args.fermi if args.fermi is not None else (ai.number(settings['fermi_energy']) if 'fermi_energy' in settings else None)
    if ef is not None: ai.require(np.isfinite(ef),'Invalid Fermi energy');cfg['fermi_eV']=ef
    source_versions.append('Wannier90 '+release)
    cfg['evidence']={
        'magnetic_state':f'User declaration: {args.magnetism}, {args.ordering}, SOC={args.soc}; POSCAR moment rank={rank}',
        'moments':'Cartesian muB vectors appended to POSCAR; omitted triples mean zero; no moment idealization performed',
        'orbital_order':'WOUT ordered projection table' if wout and not args.orbitals and not supplied_basis else 'Explicit projection/basis metadata',
        'local_axes':'Physical Cartesian axes; nontrivial source convention explicitly selected when needed',
        'home_cells':home_note,'spin_order':cfg['spin_order'],
        'spin_frame':'Global Cartesian; source SAXIS checked when available, otherwise assumed standard projection spin frame',
        'versions':'; '.join(source_versions),
        'atomic_gauge':'Explicit audit assertion or num_iter=0 projection-derived real-harmonic gauge; physical acceptance still requires band review',
        'common_gauge':'Matching ordered atomic projections required for both scalar channels; HR basis equality is an input contract',
        'real_gauge':'Real-harmonic atomic gauge; no orbital-current order assumed'}
    # Select interpolation using explicit setting or matching wsvec metadata.
    ws=[]
    if len(hrs)==1: ai.require(args.up_wsvec is None and args.down_wsvec is None,'Unexpected channel WS files')
    else: ai.require(args.wsvec is None,'Use channel-specific WS files')
    for h,explicit_ws in zip(hrs,[args.up_wsvec,args.down_wsvec] if collinear else [args.wsvec]):
        candidate=Path(explicit_ws).resolve() if explicit_ws else h.with_name(re.sub(r'_hr\.dat.*$', '_wsvec.dat', h.name))
        ws.append(candidate if candidate.is_file() else None)
        ai.require(not explicit_ws or ws[-1] is not None,'Explicit WS file does not exist')
    ws_setting=ai.truth(settings['use_ws_distance']) if 'use_ws_distance' in settings else None
    mode=args.ws_mode
    if mode=='auto':
        if ws_setting is False: mode='raw'
        elif all(p is not None for p in ws): mode='apply'
        else:
            ai.require(ws_setting is not True,'WIN requests WS interpolation but matching WS file is missing')
            ai.require(not any(ws),'Only one channel has matching WS data')
            mode='raw'
    ai.require(mode!='apply' or all(ws),'Apply mode requires matching WS files')
    ai.require(mode!='raw' or not any([args.wsvec,args.up_wsvec,args.down_wsvec]),'Explicit WS files conflict with raw interpolation')
    sources=[args.poscar.resolve()]+hrs+secondary_sources+[p for p in [win,wout,outcar,args.basis_json] if p]+([p for p in ws if p] if mode=='apply' else [])
    provenance={'interface_version':VERSION,'moment_rank':rank,'moment_rows_present':explicit,'band_path_source':note,
                'interpolation':mode,'interpolation_reason':'Explicit override' if args.ws_mode!='auto' else 'WIN use_ws_distance and matching WS files; absent evidence uses documented bare-HR Fourier convention',
                'sources_sha256':{str(Path(p).resolve()):digest(p) for p in sources},'notes':[
                'Band path is a comparison path; it does not certify a DFT fit.',
                'Orbital gauge and energy zero cannot be deduced from HR alone.']}
    return cfg,hrs,ws,mode,provenance


def execute(args):
    cfg,hrs,ws,mode,provenance=prepare(args)
    out=args.out.resolve();ai.require(not out.exists(),'Output directory already exists; choose a fresh directory')
    out.mkdir(parents=True);case=out/'case.json';case.write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    result=out/'result'
    if len(hrs)==2:
        command=[sys.executable,str(ROOT/'symmetrize_collinear_hr.py'),'--up-hr',str(hrs[0]),'--down-hr',str(hrs[1])]
        if mode=='apply': command+=['--up-wsvec',str(ws[0]),'--down-wsvec',str(ws[1])]
    else:
        runner='symmetrize_hr.py' if not cfg['magnetic'] else ('symmetrize_ssg_hr.py' if cfg['target_symmetry']=='SSG' else 'symmetrize_magnetic_hr.py')
        command=[sys.executable,str(ROOT/runner),'--hr',str(hrs[0])]
        if mode=='apply':command+=['--wsvec',str(ws[0])]
    command+=['--case-json',str(case),'--ws-mode',mode,'--out',str(result),'--symprec',str(args.symprec)]
    if cfg['magnetic']:command+=['--moment-tol',str(args.moment_tol)]
    if cfg.get('target_symmetry')=='SSG':command+=['--rank-tol',str(args.rank_tol)]
    # The nonmagnetic core expects all-up/all-down order. Convert losslessly at
    # this boundary and restore the user's original order in the delivered HR.
    converted=False
    if not cfg['magnetic'] and cfg['spinor'] and cfg['spin_order']=='interleaved':
        h=core.read_hr(hrs[0])
        if mode=='apply':
            from symmetrize_hr import apply_ws
            h=apply_ws(h,ws[0])
        converted_hr=out/'prepared_allup_alldown_hr.dat'
        core.write_hr(converted_hr,core.reorder(h,True,inverse=True))
        cfg['spin_order']='all-up-then-all-down';case.write_text(json.dumps(cfg,indent=2)+'\n')
        command[command.index('--hr')+1]=str(converted_hr)
        command[command.index('--ws-mode')+1]='raw'
        if '--wsvec' in command:
            index=command.index('--wsvec');del command[index:index+2]
        converted=True
    (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    if args.prepare_only:
        print('Prepared inputs:',out);return
    subprocess.run(command,check=True)
    if converted:
        delivered=core.reorder(core.read_hr(result/'wannier90_hr_sym.dat'),True)
        path=out/'wannier90_hr_sym.dat';core.write_hr(path,delivered,'Symmetrized HR restored to input interleaved spin order; weights one')
        provenance['delivered_spin_order']='interleaved';provenance['delivered_hr_sha256']=digest(path)
    else:
        provenance['delivered_spin_order']=cfg['spin_order']
        for name in (['wannier90_up_hr_sym.dat','wannier90_down_hr_sym.dat'] if len(hrs)==2 else ['wannier90_hr_sym.dat']):
            shutil.copyfile(result/name,out/name)
    provenance['delivered_files']={p.name:digest(p) for p in out.glob('*hr_sym.dat')}
    # Export human/machine-readable operations also for the nonmagnetic branch.
    if not cfg['magnetic']:
        basis=core.read_basis(result);ops=core.operations(basis,args.symprec)
        from magnetic import MagneticOperation, metadata, TSPIN
        from collinear import _full_mapping
        records=[]
        for anti in (False,True):
            for op in ops:
                mats=[d@(np.kron(np.eye(len(d)//2),TSPIN) if basis.spinor else np.eye(len(d))) if anti else d for d in op.matrices]
                item=MagneticOperation(core.Operation(op.rotation,op.translation,op.mapping,op.shifts,mats),anti,_full_mapping(basis,op.rotation,op.translation,args.symprec),0.)
                record=metadata([item])[0];record['index']=len(records);records.append(record)
        (result/'operations.json').write_text(json.dumps({'matrix_basis':'orbital-major interleaved spin for SOC; scalar otherwise','operations':records},indent=2)+'\n')
    ai.require(all(digest(path)==sha for path,sha in provenance['sources_sha256'].items()),'An input changed during execution')
    provenance['status']='completed';(out/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Completed. Symmetry operations, HR and before/after bands:',result)
    print('Delivered HR files:', ', '.join(str(out/name) for name in provenance['delivered_files']))


if __name__=='__main__':
    try: execute(parser().parse_args())
    except (ValueError,OSError,KeyError,IndexError,TypeError,subprocess.CalledProcessError) as error:
        print('ERROR:',error,file=sys.stderr);raise SystemExit(2)
