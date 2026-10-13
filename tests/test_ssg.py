"""Independent physical invariants for noncollinear NSOC SSG projection."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'wannsym_core'))
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
import wannsym as w
import magnetic as msg
import ssg


def fixture(coplanar=True):
    if coplanar:
        lattice=np.array([[4.,-2.,0.],[0.,np.sqrt(12),0.],[0.,0.,5.]])
        pos=np.array([[.2,.4,0.],[-.4,-.2,0.],[.2,-.2,0.]])
        moments=np.array([[1.,0.,0.],[-.5,np.sqrt(3)/2,0.],[-.5,-np.sqrt(3)/2,0.]])
    else:
        lattice=4*np.eye(3)
        moments=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)/np.sqrt(3)
        pos=.17*moments
    n=len(pos)
    basis=w.Basis(lattice,pos,['X']*n,list(range(n)),[['s']]*n,np.tile(np.eye(3),(n,1,1)),True,[2]*n,[slice(2*i,2*i+2) for i in range(n)],2*n)
    return basis,moments


def onsite(basis,moments):
    matrix=np.zeros((basis.n,basis.n),complex)
    for i,s in enumerate(basis.slices): matrix[s,s]=np.einsum('i,ijk->jk',moments[i],msg.PAULI)
    return {(0,0,0):matrix}


class SSGTests(unittest.TestCase):
    def test_coplanar_spin_only_mirror_and_independent_spin_rotation(self):
        b,m=fixture(); ops=ssg.select_operations(b,m)
        mirror=np.diag([1.,1.,-1.])
        only=[g for g in ops if np.array_equal(g.op.rotation,np.eye(3,dtype=int))
              and np.linalg.norm(g.op.translation-np.rint(g.op.translation))<1e-8]
        self.assertEqual(len(only),2)
        self.assertTrue(any(g.antiunitary and np.allclose(g.spin_rotation,mirror) for g in only))
        self.assertEqual(sum(g.antiunitary for g in ops)*2,len(ops))
        self.assertTrue(any(not np.allclose(g.spin_rotation,np.linalg.det(g.op.rotation)*b.lattice@g.op.rotation@np.linalg.inv(b.lattice)) for g in ops))
        h=onsite(b,m); out=ssg.project(h,b,ops)
        np.testing.assert_allclose(out[(0,0,0)],h[(0,0,0)],atol=1e-12)
        self.assertGreater(ssg.residuals(out,b,ops)['pure_TR_difference_eV'],1.)
        # The pure spin mirror alone removes out-of-plane sigma_z exchange,
        # while retaining the in-plane complex sigma_y contribution.
        contaminated={r:v+.4*np.kron(np.eye(len(m)),msg.PAULI[2]) for r,v in h.items()}
        projected=ssg.project(contaminated,b,only)
        np.testing.assert_allclose(projected[(0,0,0)],h[(0,0,0)],atol=1e-12)
        self.assertGreater(abs(projected[(0,0,0)].imag).max(),.5)

    def test_noncoplanar_unique_spin_maps_no_extra_stabilizer(self):
        b,m=fixture(False)
        rotation=Rotation.from_rotvec([.23,-.41,.17]).as_matrix(); m=m@rotation.T
        ops=ssg.select_operations(b,m)
        only=[g for g in ops if np.array_equal(g.op.rotation,np.eye(3,dtype=int))
              and np.linalg.norm(g.op.translation-np.rint(g.op.translation))<1e-8]
        self.assertEqual(len(only),1)
        self.assertTrue(any(g.antiunitary for g in ops))
        self.assertTrue(all(g.moment_rank==3 for g in ops))
        h=onsite(b,m)
        np.testing.assert_allclose(ssg.project(h,b,ops)[(0,0,0)],h[(0,0,0)],atol=1e-12)

    def test_semilinear_projection_against_independent_bloch_average(self):
        b,m=fixture(); b.positions[0]+=np.array([1,-1,0])
        ops=ssg.select_operations(b,m)
        rng=np.random.default_rng(417)
        a=rng.normal(size=(b.n,b.n))+1j*rng.normal(size=(b.n,b.n))
        t=rng.normal(size=(b.n,b.n))+1j*rng.normal(size=(b.n,b.n))
        h={(0,0,0):(a+a.conj().T)/2,(1,-1,0):t,(-1,1,0):t.conj().T}
        out=ssg.project(h,b,ops)
        self.assertLess(ssg.difference(out,ssg.project(out,b,ops),b.n),1e-11)
        ks=rng.uniform(-.5,.5,(4,3))
        self.assertLess(ssg.bloch_residual(out,b,ops,ks),1e-11)
        for q in ks:
            expected=np.zeros((b.n,b.n),complex)
            for g in ops:
                source_k=(-1 if g.antiunitary else 1)*q@g.op.rotation
                u=ssg.bloch_matrix(b,g,q); source_h=ssg.hk(h,source_k)
                expected+=u@(source_h.conj() if g.antiunitary else source_h)@u.conj().T/len(ops)
            np.testing.assert_allclose(ssg.hk(out,q),expected,atol=1e-11)

    def test_spin_frame_covariance(self):
        b,m=fixture(); ops=ssg.select_operations(b,m)
        rotation=Rotation.from_rotvec([.31,.27,-.12]).as_matrix()
        rotated_ops=ssg.select_operations(b,m@rotation.T)
        v=np.kron(np.eye(len(m)),msg.spin_matrix(rotation))
        rng=np.random.default_rng(11); a=rng.normal(size=(b.n,b.n))+1j*rng.normal(size=(b.n,b.n))
        h={(0,0,0):(a+a.conj().T)/2}
        out=ssg.project(h,b,ops)
        transformed=ssg.project({r:v@a@v.conj().T for r,a in h.items()},b,rotated_ops)
        for r,a in out.items(): np.testing.assert_allclose(transformed[r],v@a@v.conj().T,atol=1e-11)

    def test_reject_collinear_zero_and_incompatible_magnitudes(self):
        b,m=fixture()
        for bad in [np.zeros_like(m),np.tile([0.,0.,1.],(len(m),1))]:
            with self.assertRaisesRegex(ValueError,'rank-2 or rank-3'): ssg.select_operations(b,bad)
        # Distinct lengths exclude site-exchanging operations rather than
        # normalizing away physically relevant moment magnitudes.
        m=np.array([[1.,0.,0.],[0.,2.,0.],[-3.,0.,0.]])
        ops=ssg.select_operations(b,m,moment_tol=1e-6)
        self.assertTrue(all(g.full_mapping==[0,1,2] for g in ops))


if __name__=='__main__': unittest.main()
