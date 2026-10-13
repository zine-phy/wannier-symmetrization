"""Magnetic-space-group spinor projection in an audited atomic Wannier gauge.

Internal order: orbital-major interleaved up/down, one global Cartesian spin frame.
No pure time-reversal or real-hopping projection is imposed on an ordered magnet.
"""
from dataclasses import dataclass
import numpy as np
import spglib
from scipy.spatial.transform import Rotation
import wannsym as w
from collinear import _full_mapping, _hermitize

VERSION = '0.1.0'
TSPIN = np.array([[0, 1], [-1, 0]], complex)
PAULI = np.array([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],complex)

@dataclass
class MagneticOperation:
    op: w.Operation
    antiunitary: bool
    full_mapping: list
    moment_error: float


def spin_matrix(proper):
    qx,qy,qz,qw = Rotation.from_matrix(proper).as_quat()
    u = qw*np.eye(2)-1j*np.einsum('i,ijk->jk',[qx,qy,qz],PAULI)
    for a in range(3):
        w.require(np.allclose(u@PAULI[a]@u.conj().T,np.einsum('i,ijk->jk',proper[:,a],PAULI),atol=1e-7), 'Spin representation convention mismatch')
    return u


def select_operations(basis,moments,symprec=1e-5,moment_tol=1e-3):
    w.require(basis.spinor,'MSG projection requires a complete spinor basis')
    moments=np.asarray(moments,float)
    w.require(moments.shape==(len(basis.positions),3) and np.isfinite(moments).all(),'Supply Cartesian vector moments for all structural atoms')
    w.require(symprec>0 and moment_tol>0,'Invalid tolerances')
    w.require(np.linalg.norm(moments,axis=1).max()>moment_tol,'Use the nonmagnetic runner for zero moments')
    species={s:i+1 for i,s in enumerate(dict.fromkeys(basis.symbols))}
    cell=(basis.lattice.T,basis.positions,[species[s] for s in basis.symbols])
    sym=spglib.get_symmetry(cell,symprec=symprec)
    w.require(sym is not None,'No structural operations')
    lookup={site:i for i,site in enumerate(basis.atom_indices)}
    ops=[]; tol=max(5*symprec,1e-8)
    for rot,trans in zip(sym['rotations'],sym['translations']):
        full=_full_mapping(basis,rot,trans,tol)
        cart=basis.lattice@rot@np.linalg.inv(basis.lattice)
        w.require(np.allclose(cart.T@cart,np.eye(3),atol=1e-7,rtol=0),'Nonorthogonal spatial operation')
        axial=np.linalg.det(cart)*cart
        errors=[float(np.linalg.norm(moments[full]-sign*(moments@axial.T),axis=1).max()) for sign in [1,-1]]
        accepted=[i for i,e in enumerate(errors) if e<=moment_tol]
        w.require(len(accepted)<=1,'Ambiguous unitary/antiunitary assignment')
        if not accepted: continue
        anti=bool(accepted[0]); spin=spin_matrix(axial)
        if anti: spin=spin@TSPIN
        mapping=[]; shifts=[]; matrices=[]
        for i,site in enumerate(basis.atom_indices):
            target=full[site]
            w.require(target in lookup,'Projected sites do not close under MSG')
            j=lookup[target]
            w.require(basis.orbitals[i]==basis.orbitals[j],'Mapped orbital shells differ')
            local=basis.axes[j].T@cart@basis.axes[i]
            orbital=np.zeros((basis.sizes[i]//2,)*2,complex); start=0
            for shell in basis.orbitals[i]:
                size=w.DIMS[shell]
                orbital[start:start+size,start:start+size]=w.get_any_rot_orb_twostep(shell,local)
                start+=size
            d=np.kron(orbital,spin)
            w.require(np.allclose(d.conj().T@d,np.eye(len(d)),atol=1e-7,rtol=0),'Orbital subspace is not closed/unitary')
            mapping.append(j); shifts.append(np.rint(rot@basis.positions[site]+trans-basis.positions[target]).astype(int)); matrices.append(d)
        ops.append(MagneticOperation(w.Operation(rot,np.asarray(trans),mapping,np.asarray(shifts),matrices),anti,full,errors[int(anti)]))
    validate_group(basis,ops,tol)
    return ops


def validate_group(basis,ops,tol=1e-7):
    w.require(any(not g.antiunitary and np.array_equal(g.op.rotation,np.eye(3,dtype=int)) and np.linalg.norm(basis.lattice@(g.op.translation-np.rint(g.op.translation)))<tol for g in ops),'Missing unitary identity')
    nu=sum(not g.antiunitary for g in ops); na=len(ops)-nu
    w.require(na in (0,nu),'Antiunitary set is not an index-two coset')
    for a in ops:
        for b in ops:
            rot=a.op.rotation@b.op.rotation
            trans=a.op.rotation@b.op.translation+a.op.translation
            matches=[c for c in ops if c.antiunitary==(a.antiunitary ^ b.antiunitary) and np.array_equal(c.op.rotation,rot) and np.linalg.norm(basis.lattice@(trans-c.op.translation-np.rint(trans-c.op.translation)))<tol]
            w.require(len(matches)==1,'Moment-filtered MSG is not closed; inspect tolerances and moments')
            c=matches[0]; phase=None
            w.require([a.full_mapping[j] for j in b.full_mapping]==c.full_mapping,'Full-site composition failure')
            for i,j in enumerate(b.op.mapping):
                w.require(a.op.mapping[j]==c.op.mapping[i],'Projected-site composition failure')
                right=b.op.matrices[i].conj() if a.antiunitary else b.op.matrices[i]
                composed=a.op.matrices[j]@right; ref=c.op.matrices[i]
                if phase is None: phase=np.vdot(ref,composed)/np.vdot(ref,ref)
                w.require(abs(abs(phase)-1)<1e-7 and np.allclose(composed,phase*ref,atol=2e-7,rtol=0),'Semilinear double-group representation composition failure')
            shifts=b.op.shifts@a.op.rotation.T+a.op.shifts[b.op.mapping]-c.op.shifts
            w.require(np.all(shifts==np.rint(trans-c.op.translation).astype(int)),'Home-cell composition failure')


def validate_hr(h,basis):
    w.require(bool(h),'Empty HR')
    for r,m in h.items():
        w.require(len(r)==3 and np.array_equal(r,np.rint(r)),'Noninteger R')
        w.require(m.shape==(basis.n,basis.n) and np.isfinite(m).all(),'Invalid spinor HR dimension or entries')


def action(h,basis,item):
    op=item.op; output={}
    keys=list(h); vectors=np.asarray(keys); matrices=np.asarray([h[r] for r in keys])
    if item.antiunitary: matrices=matrices.conj()
    rotated=vectors@op.rotation.T
    for i,si in enumerate(basis.slices):
        ti=basis.slices[op.mapping[i]]
        for j,sj in enumerate(basis.slices):
            tj=basis.slices[op.mapping[j]]
            blocks=op.matrices[i]@matrices[:,si,sj]@op.matrices[j].conj().T
            for r,block in zip(rotated+op.shifts[j]-op.shifts[i],blocks):
                if not np.any(block): continue
                key=tuple(map(int,r))
                if key not in output: output[key]=np.zeros((basis.n,basis.n),complex)
                output[key][ti,tj]+=block
    return output or {(0,0,0):np.zeros((basis.n,basis.n),complex)}


def difference(a,b,n):
    zero=np.zeros((n,n),complex)
    return max(float(abs(a.get(r,zero)-b.get(r,zero)).max()) for r in set(a)|set(b))


def project(h,basis,ops):
    validate_hr(h,basis)
    w.require(bool(ops),'Empty MSG')
    output={}
    for item in ops:
        for r,m in action(h,basis,item).items():
            if r not in output: output[r]=np.zeros((basis.n,basis.n),complex)
            output[r]+=m/len(ops)
    return _hermitize(output,basis.n)


def residuals(h,basis,ops):
    validate_hr(h,basis)
    vals=[difference(h,action(h,basis,g),basis.n) for g in ops]
    u=np.kron(np.eye(basis.n//2),TSPIN)
    return {'unitary_max_eV':max([v for v,g in zip(vals,ops) if not g.antiunitary],default=0.),
            'antiunitary_max_eV':max([v for v,g in zip(vals,ops) if g.antiunitary],default=0.),
            'operation_residuals_eV':vals,
            'hermiticity_max_eV':2*difference(h,_hermitize(h,basis.n),basis.n),
            'pure_TR_difference_eV':max(float(abs(m-u@m.conj()@u.conj().T).max()) for m in h.values())}


def hk(h,k):
    return sum(m*np.exp(2j*np.pi*np.dot(k,r)) for r,m in h.items())


def bloch_matrix(basis,item,target_k):
    u=np.zeros((basis.n,basis.n),complex)
    for i,s in enumerate(basis.slices):
        u[basis.slices[item.op.mapping[i]],s]=np.exp(-2j*np.pi*np.dot(target_k,item.op.shifts[i]))*item.op.matrices[i]
    return u


def bloch_residual(h,basis,ops,ks):
    residual=0.
    for k in ks:
        original=hk(h,k)
        for item in ops:
            q=(-1 if item.antiunitary else 1)*np.asarray(k)@np.linalg.inv(item.op.rotation)
            u=bloch_matrix(basis,item,q)
            expected=u@(original.conj() if item.antiunitary else original)@u.conj().T
            residual=max(residual,float(abs(hk(h,q)-expected).max()))
    return residual


def metadata(ops):
    return [{'index':i,'antiunitary':g.antiunitary,'rotation_fractional':g.op.rotation.tolist(),'translation_fractional':g.op.translation.tolist(),
             'all_atom_map_zero_based':g.full_mapping,'projected_site_map_zero_based':g.op.mapping,'home_shifts':g.op.shifts.tolist(),
             'moment_error_muB':g.moment_error,'matrices_real':[d.real.tolist() for d in g.op.matrices],'matrices_imag':[d.imag.tolist() for d in g.op.matrices]} for i,g in enumerate(ops)]
