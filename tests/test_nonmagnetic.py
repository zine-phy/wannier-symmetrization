"""Analytic and file-level regressions for the nonmagnetic Python 3 port."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'scripts' / 'wannsym_core'))
import wannsym as w


def inputs(path,spinor=False,cubic=False,orbitals='s',two_sites=False):
    if cubic:
        lattice=np.eye(3); positions=[[0,0,0,'Si']]
    else:
        lattice=np.array([[3.1,0,0],[.31,3.7,0],[.23,.41,4.3]])
        positions=[[.13,.21,.31,'Si'],[.37,.11,.63,'O'],[.19,.73,.42,'Se']]
    text='Test\n1\n'+'\n'.join(' '.join(map(str,row)) for row in lattice)+'\n#\n#\n'+str(len(positions))+'\n'
    text+='\n'.join(' '.join(map(str,row)) for row in positions)+'\n'
    (path/'poscar.in').write_text(text)
    count=2 if two_sites else 1
    (path/'wann.in').write_text('#\n'*23+str(count)+'\n'+('T' if spinor else 'F')+'\n'+''.join(f'{i+1} {i+1} {orbitals}\n' for i in range(count)))
    (path/'locaxis.in').write_text('#\n'*19+'0\n')


def cli(path):
    result=subprocess.run([sys.executable,str(w.ROOT/'wannsym.py'),str(path)],capture_output=True,text=True)
    if result.returncode: raise AssertionError(result.stdout+'\n'+result.stderr)
    return w.read_hr(path/'wannier90_hr_sym.dat'), json.loads((path/'wannier90_hr_sym.dat.report.json').read_text())


class NonmagneticTests(unittest.TestCase):
    def test_spinless_missing_star(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); inputs(p,cubic=True)
            h={(0,0,0):np.array([[1.]]), (1,0,0):np.array([[.3]]),(-1,0,0):np.array([[.3]])}
            w.write_hr(p/'wannier90_hr.dat',h)
            out,report=cli(p)
            self.assertEqual(report['operations'],48)
            self.assertEqual(len(out),7)
            for r,m in out.items(): self.assertAlmostEqual(m[0,0].real,1. if r==(0,0,0) else .3,places=12)
            k=np.array([[.17,.29,.31]])
            expected=1+.6*np.cos(2*np.pi*k).sum()
            self.assertAlmostEqual(w.bands(out,k)[0,0],expected,places=12)

    def test_pure_imaginary_soc_and_spin_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); inputs(p,spinor=True,two_sites=True)
            # External order: orbital1 up, orbital2 up, orbital1 down, orbital2 down.
            onsite=np.diag([1.,2.,1.,2.]).astype(complex)
            hop=1j*np.diag([.2,.3,-.2,-.3])
            h={(0,0,0):onsite,(1,0,0):hop,(-1,0,0):hop.conj().T}
            w.write_hr(p/'wannier90_hr.dat',h)
            out,report=cli(p)
            self.assertEqual(report['operations'],1)
            self.assertEqual(len(report['pure_imaginary_R_preserved']),2)
            for r in h: np.testing.assert_allclose(out[r],h[r],atol=1e-13)
            k=np.array([[.25,0,0]])
            np.testing.assert_allclose(w.bands(out,k)[0],np.sort([.6,1.4,1.4,2.6]),atol=1e-12)

    def test_atomic_p_soc_cubic(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); inputs(p,spinor=True,cubic=True,orbitals='p')
            # Cartesian vector representation (px,py,pz), converted to (pz,px,py).
            L=[np.array([[0,0,0],[0,0,-1j],[0,1j,0]]),
               np.array([[0,0,1j],[0,0,0],[-1j,0,0]]),
               np.array([[0,-1j,0],[1j,0,0],[0,0,0]])]
            sigma=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
            order=[2,0,1]
            matrix=sum(np.kron(l[np.ix_(order,order)],s) for l,s in zip(L,sigma))*.1
            h=w.reorder({(0,0,0):matrix},True,inverse=True)
            w.write_hr(p/'wannier90_hr.dat',h)
            out,report=cli(p)
            np.testing.assert_allclose(out[(0,0,0)],h[(0,0,0)],atol=1e-12)
            np.testing.assert_allclose(np.linalg.eigvalsh(out[(0,0,0)]),[-.2,-.2,.1,.1,.1,.1],atol=1e-12)
            self.assertLess(max(report['residuals'].values()),1e-12)

    def test_hr_permutation_degeneracies_and_non_soc_tr(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); inputs(p)
            # Deliberately nonlexicographic R order with distinct weights.
            rows=[((1,0,0),3,.6+0.9j),((0,0,0),1,1.0),((-1,0,0),2,.4-0.6j)]
            text='Permuted HR\n1\n3\n3 1 2\n'
            for r,deg,z in rows:
                z=complex(z);text+=f'{r[0]} {r[1]} {r[2]} 1 1 {z.real} {z.imag}\n'
            (p/'wannier90_hr.dat').write_text(text)
            out,report=cli(p)
            self.assertAlmostEqual(out[(1,0,0)][0,0],.2)
            self.assertAlmostEqual(out[(-1,0,0)][0,0],.2)
            self.assertEqual(report['spinor'],False)

    def test_invalid_hr_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); inputs(p)
            (p/'wannier90_hr.dat').write_text('Bad\n1\n1\n1\n')
            result=subprocess.run([sys.executable,str(w.ROOT/'wannsym.py'),str(p)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((p/'wannier90_hr_sym.dat').exists())
            w.write_hr(p/'wannier90_hr.dat',{(0,0,0):np.eye(1)})
            cli(p)
            before=(p/'wannier90_hr_sym.dat').read_bytes()
            result=subprocess.run([sys.executable,str(w.ROOT/'wannsym.py'),str(p)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(before,(p/'wannier90_hr_sym.dat').read_bytes())


if __name__=='__main__': unittest.main(verbosity=2)
