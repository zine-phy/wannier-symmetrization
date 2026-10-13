from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'wannsym_core'))
import unittest
import numpy as np
import wannsym as w
import magnetic as m


def basis(lattice=None,positions=None,many=False,shell='s'):
    lattice=np.eye(3)*4 if lattice is None else np.asarray(lattice)
    positions=np.array([[0,0,0]] if positions is None else positions,float)
    nsite=len(positions); sizes=[2*w.DIMS[shell]]*nsite; n=sum(sizes)
    return w.Basis(lattice,positions,['X']*nsite,list(range(nsite)),[[shell] for _ in positions],np.tile(np.eye(3),(nsite,1,1)),True,sizes,[slice(i*sizes[0],(i+1)*sizes[0]) for i in range(nsite)],n)

class MagneticTests(unittest.TestCase):
    def test_ferromagnet_keeps_exchange_and_breaks_pure_TR(self):
        b=basis(); ops=m.select_operations(b,[[0,0,2.]])
        self.assertEqual(sum(g.antiunitary for g in ops),8)
        self.assertEqual(len(ops),16)
        h={(0,0,0):2*np.eye(2)+.7*m.PAULI[2]}
        out=m.project(h,b,ops)
        np.testing.assert_allclose(out[(0,0,0)],h[(0,0,0)],atol=1e-12)
        self.assertGreater(m.residuals(out,b,ops)['pure_TR_difference_eV'],1.)

    def test_atomic_p_SOC_plus_exchange(self):
        b=basis(shell='p'); ops=m.select_operations(b,[[0,0,1.]])
        l=np.array([[[0,0,0],[0,0,-1j],[0,1j,0]],[[0,0,1j],[0,0,0],[-1j,0,0]],[[0,-1j,0],[1j,0,0],[0,0,0]]])
        order=[2,0,1]; l=l[:,order][:,:,order]
        onsite=.31*sum(np.kron(l[i],m.PAULI[i]) for i in range(3))+.8*np.kron(np.eye(3),m.PAULI[2])
        h={(0,0,0):onsite}; out=m.project(h,b,ops)
        np.testing.assert_allclose(out[(0,0,0)],onsite,atol=1e-12)
        self.assertGreater(abs(out[(0,0,0)].imag).max(),.1)
        self.assertLess(m.residuals(out,b,ops)['antiunitary_max_eV'],1e-12)

    def test_unitary_only_triclinic(self):
        b=basis(lattice=[[3.2,.2,.3],[0,4.1,.4],[0,0,5.3]])
        ops=m.select_operations(b,[[.31,.72,1.1]])
        self.assertEqual(len(ops),2); self.assertFalse(any(g.antiunitary for g in ops))
        h={(0,0,0):np.eye(2)+m.PAULI[1]}
        np.testing.assert_allclose(m.project(h,b,ops)[(0,0,0)],h[(0,0,0)],atol=1e-12)

    def test_AF_translation_semilinear_independent_k_average(self):
        b=basis(lattice=[[3.2,.2,.3],[0,4.1,.4],[0,0,5.3]],positions=[[0,0,0],[.5,0,0]])
        ops=m.select_operations(b,[[0,0,1.],[0,0,-1.]])
        self.assertTrue(any(g.antiunitary and np.linalg.norm(g.op.translation)>.1 for g in ops))
        rng=np.random.default_rng(1); a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); t=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
        h={(0,0,0):(a+a.conj().T)/2,(1,-1,0):t,(-1,1,0):t.conj().T}
        out=m.project(h,b,ops)
        self.assertLess(m.difference(out,m.project(out,b,ops),b.n),1e-12)
        self.assertLess(m.bloch_residual(out,b,ops,rng.uniform(-.5,.5,(5,3))),1e-11)
        for q in rng.uniform(-.5,.5,(4,3)):
            expected=np.zeros((4,4),complex)
            for g in ops:
                k=(-1 if g.antiunitary else 1)*q@g.op.rotation
                u=m.bloch_matrix(b,g,q); original=m.hk(h,k)
                expected+=u@(original.conj() if g.antiunitary else original)@u.conj().T/len(ops)
            np.testing.assert_allclose(m.hk(out,q),expected,atol=1e-11)

    def test_pure_imaginary_R_survives_file(self):
        import tempfile
        from pathlib import Path
        b=basis(lattice=[[3.2,.2,.3],[0,4.1,.4],[0,0,5.3]])
        ops=m.select_operations(b,[[0,1,0]])
        h={(0,0,0):np.eye(2),(1,0,0):m.PAULI[1],(-1,0,0):m.PAULI[1]}
        out=m.project(h,b,ops)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'hr.dat'; w.write_hr(p,out); back=w.read_hr(p)
            np.testing.assert_allclose(back[(1,0,0)],m.PAULI[1],atol=1e-12)

    def test_all_structural_moments_and_invalid_input(self):
        b=basis(positions=[[0,0,0],[.2,.3,.4]])
        b.symbols=['X','Y']; b.atom_indices=[0]; b.orbitals=[['s']]; b.axes=b.axes[:1]; b.sizes=[2]; b.slices=[slice(0,2)]; b.n=2
        ops=m.select_operations(b,[[0,0,1],[1,2,3]])
        self.assertFalse(any(g.antiunitary for g in ops))
        with self.assertRaises(ValueError): m.select_operations(b,[[0,0,0],[0,0,0]])
        with self.assertRaises(ValueError): m.select_operations(b,[1,2])

if __name__=='__main__': unittest.main()
