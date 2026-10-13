#!/usr/bin/env python3
"""Generate small synthetic HR examples; no material data is required."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'wannsym_core'))
import wannsym as w


def case(shell='s'):
    return {
        'magnetic': False, 'spinor': False, 'spin_order': 'scalar',
        'axes_convention': 'physical-cartesian-columns',
        'lattice_A': (4*np.eye(3)).tolist(),
        'atoms': [{'element': 'X', 'fractional': [0, 0, 0]}],
        'sites': [{'atom': 1, 'shells': [shell], 'x': [1, 0, 0], 'z': [0, 0, 1]}],
        'evidence': {key: 'Explicit synthetic atomic basis defined by make_demo.py'
                     for key in ['magnetic_state', 'orbital_order', 'local_axes', 'home_cells',
                                 'spin_order', 'versions', 'common_gauge', 'real_gauge',
                                 'spin_frame', 'atomic_gauge', 'moments']},
        'fermi_eV': 0.,
        'band_path': [{'start': [0, 0, 0], 'end': [.5, 0, 0], 'labels': ['Gamma', 'X'], 'points': 21},
                      {'start': [.5, 0, 0], 'end': [.5, .5, 0], 'labels': ['X', 'M'], 'points': 21}]
    }


def hopping(onsite):
    h = {(0, 0, 0): np.asarray(onsite, complex)}
    for axis in range(3):
        for sign in [-1, 1]:
            r = [0, 0, 0]; r[axis] = sign
            h[tuple(r)] = .1*np.eye(len(onsite))
    return h



def write_metadata(directory, cfg, models):
    """Write portable POSCAR/WIN/WOUT fixtures for the automatic interface."""
    directory.mkdir()
    atoms=cfg['atoms']; lattice=np.array(cfg['lattice_A'])
    symbols=list(dict.fromkeys(a['element'] for a in atoms))
    moments=cfg.get('moments_cartesian_muB', [[0.,0.,m] for m in cfg.get('moments_muB',[0.]*len(atoms))])
    lines=['Synthetic metadata fixture', '1.0']+[' '.join(map(str,row)) for row in lattice]
    lines+=[' '.join(symbols),' '.join(str(sum(a['element']==s for a in atoms)) for s in symbols),'Direct']
    lines+=[' '.join(map(str,a['fractional']+m)) for a,m in zip(atoms,moments)]
    (directory/'POSCAR').write_text('\n'.join(lines)+'\n')
    dim=len(next(iter(next(iter(models.values())).values())))
    win=[f'num_wann = {dim}', 'num_iter = 0', 'spinors = '+str(cfg['spinor']).lower(), 'use_ws_distance = false', 'fermi_energy = 0', 'begin unit_cell_cart']
    win+=[' '.join(map(str,row)) for row in lattice]+['end unit_cell_cart','begin kpoint_path']
    for seg in cfg['band_path']:
        win.append(' '.join([seg['labels'][0],*map(str,seg['start']),seg['labels'][1],*map(str,seg['end'])]))
    win+=['end kpoint_path']
    (directory/'wannier90.win').write_text('\n'.join(win)+'\n')
    rows=[]; centers=[]
    for site in cfg['sites']:
        position=atoms[site['atom']-1]['fractional']
        for shell in site['shells']:
            ell='spdf'.index(shell)
            for mr in range(1,2*ell+2):
                rows.append([*position,ell,mr,1,*site['z'],*site['x'],1.])
                centers.append(np.array(position)@lattice)
    if cfg['spinor']:
        if cfg['spin_order']=='interleaved':
            rows=[row for row in rows for _ in range(2)];centers=[row for row in centers for _ in range(2)]
        else: rows=rows+rows;centers=centers+centers
    table=['Release: synthetic-fixture', 'PROJECTIONS']+['| '+' '.join(map(str,row))+' |' for row in rows]+['+----------------+']
    table+=['WF centre and spread %d ( %s ) 0.0'%(i+1,', '.join(map(str,row))) for i,row in enumerate(centers)]
    for seed,model in models.items():
        w.write_hr(directory/(seed+'_hr.dat'), model, 'Synthetic fixture; not a material model')
        (directory/(seed+'.wout')).write_text('\n'.join(table)+'\n')


def generate(out):
    out.mkdir(parents=True, exist_ok=False)
    scalar = case()
    soc = case('p'); soc.update(spinor=True, spin_order='all-up-then-all-down')
    collinear = case(); collinear.update(magnetic=True, spin_order='separate-up-down',
                                        moments_muB=[1.], real_gauge=True)
    magnetic = case('p'); magnetic.update(magnetic=True, spinor=True, target_symmetry='MSG',
        source_soc=True, spin_order='interleaved', spin_frame='global-cartesian',
        moments_cartesian_muB=[[0., 0., 1.]])
    # Cartesian p representation transformed to the core's (pz, px, py) order.
    angular = np.array([[[0,0,0],[0,0,-1j],[0,1j,0]],
                        [[0,0,1j],[0,0,0],[-1j,0,0]],
                        [[0,-1j,0],[1j,0,0],[0,0,0]]])
    order = [2, 0, 1]; angular = angular[:, order][:, :, order]
    sigma = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]])
    ls = .31*sum(np.kron(angular[i], sigma[i]) for i in range(3))
    models = {'scalar': hopping([[1.]]),
              'soc': w.reorder(hopping(ls), True, inverse=True),
              'up': hopping([[.7]]), 'down': hopping([[-.7]]),
              'magnetic': hopping(ls + .8*np.kron(np.eye(3), sigma[2]))}
    # Planar triangular texture: full non-SOC spinor model, not ISPIN=2.
    ssg_case = case(); ssg_case.update(magnetic=True, spinor=True, source_soc=False,
        target_symmetry='SSG', spin_order='interleaved', spin_frame='global-cartesian',
        lattice_A=[[4.,0.,0.],[-2.,float(np.sqrt(12)),0.],[0.,0.,5.]],
        atoms=[{'element':'X','fractional':pos} for pos in [[.2,.4,0.],[-.4,-.2,0.],[.2,-.2,0.]]],
        sites=[{'atom':i+1,'shells':['s'],'x':[1,0,0],'z':[0,0,1]} for i in range(3)],
        moments_cartesian_muB=[[1.,0.,0.],[-.5,float(np.sqrt(3)/2),0.],[-.5,float(-np.sqrt(3)/2),0.]])
    exchange=np.zeros((6,6),complex)
    for i,moment in enumerate(ssg_case['moments_cartesian_muB']):
        exchange[2*i:2*i+2,2*i:2*i+2]=np.einsum('i,ijk->jk',moment,sigma)
    # z hopping is invariant under the hexagonal structural group.
    models['ssg']={(0,0,0):exchange,(0,0,1):.1*np.eye(6),(0,0,-1):.1*np.eye(6)}
    for name, cfg in [('scalar',scalar),('soc',soc),('magnetic',magnetic),('ssg',ssg_case)]:
        write_metadata(out/name, cfg, {'wannier90':models[name]})
    write_metadata(out/'collinear', collinear, {'wannier90.up':models['up'], 'wannier90.dn':models['down']})
    (out / 'ssg.json').write_text(json.dumps(ssg_case,indent=2)+'\n')
    for name, model in models.items():
        w.write_hr(out / f'{name}_hr.dat', model, 'Synthetic portability example; not a material model')
    for name, cfg in [('scalar', scalar), ('soc', soc), ('collinear', collinear), ('magnetic', magnetic)]:
        (out / f'{name}.json').write_text(json.dumps(cfg, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='Fresh directory for generated HR and JSON files')
    args = parser.parse_args()
    generate(args.out)
    print(f'Created synthetic scalar, SOC, collinear FM, magnetic SOC and noncollinear NSOC examples in {args.out}')
