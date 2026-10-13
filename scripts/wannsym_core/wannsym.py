#!/usr/bin/env python3
"""Nonmagnetic scalar/spinor HR symmetrization, ported from Changming Yue's symwann.

The upstream real-orbital and spin rotation routines are retained in lib/.
The driver uses explicit R keys, validated input, and sparse block completion.
See the skill's references/provenance.md and LICENSE for attribution.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import platform
import time
import numpy as np
import spglib
import sympy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'lib'))
from get_orb_rotmat_twostep import get_any_rot_orb_twostep
from get_euler_angle import rmat2euler
from rotate import dmat_spinor

VERSION = '1.0.0'
DIMS = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 't2g': 3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(s):
    return float(s.replace('D', 'E').replace('d', 'e'))


def read_hr(path):
    """Return physically weighted H(R), accepting arbitrary R and entry order."""
    with Path(path).open() as f:
        f.readline()
        n = int(f.readline()); nr = int(f.readline())
        require(n > 0 and nr > 0, 'HR dimensions must be positive')
        weights = []
        while len(weights) < nr:
            line = f.readline()
            require(bool(line), 'Truncated HR degeneracy list')
            weights.extend(int(x) for x in line.split())
        require(len(weights) == nr and min(weights) > 0, 'Invalid HR degeneracies')
        h = {}
        for block in range(nr):
            matrix = np.zeros((n, n), complex)
            seen = set(); r = None
            for _ in range(n*n):
                fields = f.readline().split()
                require(len(fields) == 7, 'Truncated/malformed HR matrix row')
                xyz = tuple(int(v) for v in fields[:3])
                if r is None: r = xyz
                require(xyz == r, 'R changes within an HR block')
                i, j = int(fields[3])-1, int(fields[4])-1
                require(0 <= i < n and 0 <= j < n and (i,j) not in seen,
                        'Duplicate or out-of-range HR orbital index')
                seen.add((i,j))
                matrix[i,j] = complex(number(fields[5]), number(fields[6]))
            require(r not in h, 'Duplicate HR R block')
            require(np.isfinite(matrix).all(), 'Nonfinite HR entries')
            h[r] = matrix / weights[block]
        require(not f.read().strip(), 'Unexpected trailing HR records')
    return h


def write_hr(path, h, title='Wannsym Python 3 nonmagnetic symmetrization'):
    n = next(iter(h.values())).shape[0]
    with Path(path).open('w') as f:
        f.write(f'{title}\n{n:12d}\n{len(h):12d}\n')
        weights = ['1'] * len(h)
        for start in range(0, len(weights), 15):
            f.write('    ' + '    '.join(weights[start:start+15]) + '\n')
        for r, matrix in sorted(h.items()):
            for j in range(n):
                for i in range(n):
                    z = matrix[i,j]
                    f.write(f'{r[0]:5d}{r[1]:5d}{r[2]:5d}{i+1:5d}{j+1:5d} {z.real: .16e} {z.imag: .16e}\n')


@dataclass
class Basis:
    lattice: np.ndarray                 # columns are lattice vectors
    positions: np.ndarray               # fractional site rows
    symbols: list
    atom_indices: list
    orbitals: list
    axes: np.ndarray                    # local x,y,z as columns
    spinor: bool
    sizes: list
    slices: list
    n: int


def read_basis(directory):
    directory = Path(directory)
    p = (directory/'poscar.in').read_text().splitlines()
    lattice = np.array([[number(x) for x in row.split()] for row in p[2:5]]).T
    require(lattice.shape == (3,3) and abs(np.linalg.det(lattice)) > 1e-12, 'Invalid lattice')
    nat = int(p[7].split()[0]); require(nat > 0, 'No structure atoms')
    rows = [line.split() for line in p[8:8+nat]]
    require(len(rows) == nat and all(len(row)==4 for row in rows), 'Invalid poscar.in positions')
    positions = np.array([[number(x) for x in row[:3]] for row in rows]); symbols=[row[3] for row in rows]
    w = (directory/'wann.in').read_text().splitlines()
    nw = int(w[23].split()[0]); flag=w[24].strip().upper()
    require(nw > 0 and flag in ('T','F'), 'wann.in requires positive site count and T/F')
    ids=[]; orbitals=[]
    for i,row in enumerate(w[25:25+nw]):
        fields=row.split()
        require(len(fields)>=3 and int(fields[0])==i+1, 'Projected site indices must be sequential')
        site=int(fields[1])-1
        require(0<=site<nat and site not in ids, 'Invalid/duplicate projected structural site')
        require(all(orb in DIMS for orb in fields[2:]), 'Unsupported orbital label')
        ids.append(site); orbitals.append(fields[2:])
    require(len(ids)==nw, 'Truncated wann.in')
    axes=np.tile(np.eye(3),(nw,1,1))
    a=(directory/'locaxis.in').read_text().splitlines(); count=int(a[19].split()[0])
    require(0<=count<=nw and len(a)>=20+count, 'Invalid local-axis entry count')
    used=set()
    for row in a[20:20+count]:
        fields=row.split(); require(len(fields)==7, 'Invalid locaxis.in row')
        i=int(fields[0])-1; require(0<=i<nw and i not in used, 'Invalid/duplicate local-axis site')
        used.add(i)
        z=np.array([number(v) for v in fields[1:4]]); x=np.array([number(v) for v in fields[4:7]])
        require(np.linalg.norm(x)>1e-12 and np.linalg.norm(z)>1e-12, 'Zero local axis')
        z/=np.linalg.norm(z); x/=np.linalg.norm(x)
        require(abs(x@z)<1e-7, 'Local x and z axes must be orthogonal')
        axes[i]=np.column_stack((x,np.cross(z,x),z))
    spinor=flag=='T'; sizes=[sum(DIMS[o] for o in orbs)*(2 if spinor else 1) for orbs in orbitals]
    offsets=np.cumsum([0]+sizes)
    slices=[slice(int(offsets[i]),int(offsets[i+1])) for i in range(nw)]
    require(np.isfinite(lattice).all() and np.isfinite(positions).all(), 'Nonfinite structure')
    return Basis(lattice,positions,symbols,ids,orbitals,axes,spinor,sizes,slices,int(offsets[-1]))


@dataclass
class Operation:
    rotation: np.ndarray
    translation: np.ndarray
    mapping: list
    shifts: np.ndarray
    matrices: list

    def transform_key(self, r, i, j):
        return tuple((self.rotation@np.asarray(r)+self.shifts[j]-self.shifts[i]).tolist()), self.mapping[i], self.mapping[j]


def operations(basis, symprec):
    species={s:i+1 for i,s in enumerate(dict.fromkeys(basis.symbols))}
    cell=(basis.lattice.T,basis.positions,[species[s] for s in basis.symbols])
    symmetry=spglib.get_symmetry(cell,symprec=symprec)
    require(symmetry is not None, 'spglib could not find symmetry')
    result=[]; atom_lookup={site:i for i,site in enumerate(basis.atom_indices)}
    invlat=np.linalg.inv(basis.lattice)
    for rot,trans in zip(symmetry['rotations'],symmetry['translations']):
        mapping=[]; shifts=[]; matrices=[]
        cart=basis.lattice@rot@invlat
        require(np.allclose(cart.T@cart,np.eye(3),atol=1e-7), 'Nonorthogonal spatial operation')
        if basis.spinor:
            proper=np.linalg.det(cart)*cart
            angles=rmat2euler(proper)
            spin=dmat_spinor(*angles)
            require(np.allclose(spin.conj().T@spin,np.eye(2),atol=1e-7), 'Nonunitary spin representation')
        for i,site in enumerate(basis.atom_indices):
            transformed=rot@basis.positions[site]+trans
            candidates=[]
            for target in range(len(basis.positions)):
                if basis.symbols[target]!=basis.symbols[site]: continue
                delta=transformed-basis.positions[target]
                if np.linalg.norm(basis.lattice@(delta-np.rint(delta))) < max(5*symprec,1e-8):
                    candidates.append(target)
            require(len(candidates)==1, 'Ambiguous/unmatched symmetry site mapping')
            target=candidates[0]
            require(target in atom_lookup, 'Projected atoms do not close under the space group')
            j=atom_lookup[target]; mapping.append(j)
            shifts.append(np.rint(transformed-basis.positions[target]).astype(int))
            require(basis.orbitals[i]==basis.orbitals[j], 'Mapped sites have incompatible orbital shells')
            local=basis.axes[j].T@cart@basis.axes[i]
            dim=sum(DIMS[o] for o in basis.orbitals[i]); orbital=np.zeros((dim,dim),complex); start=0
            for orb in basis.orbitals[i]:
                block=get_any_rot_orb_twostep(orb,local); size=DIMS[orb]
                orbital[start:start+size,start:start+size]=block; start+=size
            d=np.kron(orbital,spin) if basis.spinor else orbital
            require(np.allclose(d.conj().T@d,np.eye(len(d)),atol=1e-7), 'Orbital subspace not closed/unitary')
            matrices.append(d)
        require(len(set(mapping))==len(mapping),'Site map is not a permutation')
        result.append(Operation(rot.astype(int),trans,mapping,np.array(shifts),matrices))
    return result


def spin_permutation(n):
    require(n%2==0,'Spinor dimension must be even')
    return np.column_stack((np.arange(n//2),np.arange(n//2,n))).ravel()


def reorder(h, spinor, inverse=False):
    if not spinor: return {r:m.copy() for r,m in h.items()}
    p=spin_permutation(next(iter(h.values())).shape[0])
    if inverse: p=np.argsort(p)
    return {r:m[np.ix_(p,p)] for r,m in h.items()}


def prepare(h,basis):
    for r in h: require(tuple(-np.array(r)) in h,'HR R support must contain both R and -R')
    h={r:(m+h[tuple(-np.array(r))].conj().T)/2 for r,m in h.items()}
    if basis.spinor:
        u=np.kron(np.eye(basis.n//2),np.array([[0,1],[-1,0]]))
        return {r:(m+u@m.conj()@u.conj().T)/2 for r,m in h.items()}
    return {r:m.real.astype(complex) for r,m in h.items()}


def complete_and_average(h,basis,ops):
    """Retain the upstream missing atom-pair-block completion before averaging."""
    ns=len(basis.sizes); support={(r,i,j) for r in h for i in range(ns) for j in range(ns)}
    original=support.copy()
    for key in original:
        for op in ops: support.add(op.transform_key(*key))
    completed={r:m.copy() for r,m in h.items()}
    for r,i,j in sorted(support-original):
        for op in ops:
            rr,ii,jj=op.transform_key(r,i,j)
            if (rr,ii,jj) in original:
                block=op.matrices[i].conj().T@h[rr][basis.slices[ii],basis.slices[jj]]@op.matrices[j]
                if r not in completed: completed[r]=np.zeros((basis.n,basis.n),complex)
                completed[r][basis.slices[i],basis.slices[j]]=block
                break
        else: raise ValueError('Could not complete a symmetry-related hopping block')
    out={r:np.zeros((basis.n,basis.n),complex) for r in completed}
    for r,i,j in sorted(support):
        block=np.zeros((basis.sizes[i],basis.sizes[j]),complex)
        for op in ops:
            rr,ii,jj=op.transform_key(r,i,j)
            require((rr,ii,jj) in support,'Symmetry orbit is not closed')
            block+=op.matrices[i].conj().T@completed[rr][basis.slices[ii],basis.slices[jj]]@op.matrices[j]
        out[r][basis.slices[i],basis.slices[j]]=block/len(ops)
    return out, len(support-original)


def residuals(h,basis,ops):
    zero=np.zeros((basis.n,basis.n),complex)
    herm=tr=space=0.; ns=len(basis.sizes)
    u=np.kron(np.eye(basis.n//2),np.array([[0,1],[-1,0]])) if basis.spinor else None
    for r,m in h.items():
        herm=max(herm,float(np.max(abs(m-h.get(tuple(-np.array(r)),zero).conj().T))))
        tr=max(tr,float(np.max(abs(m-(u@m.conj()@u.conj().T if basis.spinor else m.conj())))))
        for i in range(ns):
            for j in range(ns):
                for op in ops:
                    rr,ii,jj=op.transform_key(r,i,j)
                    other=h.get(rr,zero)[basis.slices[ii],basis.slices[jj]]
                    delta=m[basis.slices[i],basis.slices[j]]-op.matrices[i].conj().T@other@op.matrices[j]
                    space=max(space,float(np.max(abs(delta))))
    return {'hermiticity_max_eV':herm,'time_reversal_max_eV':tr,'space_group_max_eV':space}


def bands(h,points):
    return np.array([np.linalg.eigvalsh(sum(m*np.exp(2j*np.pi*np.dot(k,r)) for r,m in h.items())) for k in points])


def run(args):
    start=time.monotonic(); directory=Path(args.directory).resolve()
    input_path=(directory/args.input).resolve(); output_path=(directory/args.output).resolve()
    require(input_path!=output_path,'Never overwrite the input HR')
    require(not output_path.exists(),'Output exists; use a new output name/directory')
    report_path=output_path.with_name(output_path.name+'.report.json')
    require(not report_path.exists(),'Report already exists')
    basis=read_basis(directory); external=read_hr(input_path)
    require(next(iter(external.values())).shape==(basis.n,basis.n),'HR and wann.in dimensions disagree')
    h=reorder(external,basis.spinor); ops=operations(basis,args.symprec)
    print(f'Wannsym {VERSION}: spinor={basis.spinor}, orbitals={basis.n}, R={len(h)}, operations={len(ops)}',flush=True)
    projected=prepare(h,basis)
    out,added=complete_and_average(projected,basis,ops)
    # Check the entire complex matrix, never its real part alone.
    retained={r:m for r,m in out.items() if np.abs(m).sum()>=args.drop_tol}
    if not retained: retained={(0,0,0):np.zeros((basis.n,basis.n),complex)}
    check=residuals(retained,basis,ops)
    require(max(check.values())<=args.validation_tol,f'Symmetry validation failed: {check}')
    output_external=reorder(retained,basis.spinor,inverse=True)
    write_hr(output_path,output_external)
    reread=read_hr(output_path)
    io_error=max(float(np.max(abs(reread[r]-m))) for r,m in output_external.items())
    points=np.vstack((np.zeros((1,3)),np.eye(3)*.5,np.random.default_rng(42).random((12,3))))
    delta=float(np.max(abs(bands(external,points)-bands(reread,points))))
    pure_imag=[list(r) for r,m in retained.items() if np.abs(m.real).sum()<1e-9 and np.abs(m.imag).sum()>=1e-9]
    report={'version':VERSION,'spinor':basis.spinor,'spin_order':'all-up-then-all-down' if basis.spinor else 'scalar',
            'num_wann':basis.n,'input_R':len(h),'output_R':len(retained),'operations':len(ops),'added_atom_pair_blocks':added,
            'symprec_Angstrom':args.symprec,'drop_tol_eV':args.drop_tol,'validation_tol_eV':args.validation_tol,
            'residuals':check,'write_read_max_eV':io_error,'generic_k_band_change_max_eV':delta,
            'pure_imaginary_R_preserved':pure_imag,'input':str(input_path),'output':str(output_path),
            'inputs_sha256':{str(p):sha256(p) for p in [input_path]+[directory/x for x in ['poscar.in','wann.in','locaxis.in']]},
            'output_sha256':sha256(output_path),'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__)]+sorted((ROOT/'lib').glob('*.py'))},
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'sympy':sympy.__version__,'spglib':spglib.__version__},
            'elapsed_seconds':time.monotonic()-start}
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    # Numeric representations for independent diagnostics; no import-time file writes.
    np.savez_compressed(str(output_path)+'.operations.npz',rotations=np.array([o.rotation for o in ops]),
        translations=np.array([o.translation for o in ops]),mappings=np.array([o.mapping for o in ops]),
        shifts=np.array([o.shifts for o in ops]))
    if (directory/'write_symmop.flag').exists():
        for index,op in enumerate(ops,1):
            d=np.zeros((basis.n,basis.n),complex)
            for i,j in enumerate(op.mapping): d[basis.slices[j],basis.slices[i]]=op.matrices[i]
            np.save(str(output_path)+f'.representation_{index}.npy',d)
    print(json.dumps(report,indent=2),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version',action='version',version=VERSION)
    parser.add_argument('directory',nargs='?',default='.')
    parser.add_argument('--input',default='wannier90_hr.dat')
    parser.add_argument('--output',default='wannier90_hr_sym.dat')
    parser.add_argument('--symprec',type=float,default=1e-5)
    parser.add_argument('--drop-tol',type=float,default=1e-12)
    parser.add_argument('--validation-tol',type=float,default=1e-7)
    args=parser.parse_args()
    try:
        require(args.symprec>0 and args.drop_tol>=0 and args.validation_tol>0,'Invalid tolerance')
        run(args)
    except (ValueError,OSError,IndexError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':
    raise SystemExit(main())
