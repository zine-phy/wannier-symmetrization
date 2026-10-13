"""Automatic input construction and full five-branch interface regression."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import auto_inputs as ai
import symmetrize as entry


class AutomaticInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='automatic wannsym ')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base=Path(cls.temp.name);cls.demo=cls.base/'demo'
        cls.env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',WANNSYM_FONT='DejaVu Serif')
        subprocess.run([sys.executable,str(ROOT/'examples/make_demo.py'),'--out',str(cls.demo)],check=True,capture_output=True,env=cls.env)

    def args(self,branch,extra=()):
        folder=self.demo/branch
        cmd=['--poscar',str(folder/'POSCAR'),'--magnetism','nonmagnetic' if branch in ('scalar','soc') else 'magnetic',
             '--ordering','noncollinear' if branch=='ssg' else 'collinear','--soc','yes' if branch in ('soc','magnetic') else 'no',
             '--points','5','--out',str(self.base/(branch+' output'))]
        if branch=='collinear':cmd+=['--up-hr',str(folder/'wannier90.up_hr.dat'),'--down-hr',str(folder/'wannier90.dn_hr.dat')]
        else:cmd+=['--hr',str(folder/'wannier90_hr.dat')]
        return entry.parser().parse_args(cmd+list(extra))

    def run_branch(self,branch):
        args=self.args(branch)
        entry.execute(args)
        out=args.out/'result'
        self.assertEqual(json.loads((out/'manifest.json').read_text())['status'],'completed')
        operations=json.loads((out/'operations.json').read_text())
        operations=operations['operations'] if isinstance(operations,dict) else operations
        self.assertTrue(operations)
        if branch in ('scalar','soc'):
            self.assertEqual(sum(op['antiunitary'] for op in operations)*2,len(operations))
            self.assertTrue(all(op['all_atom_map_zero_based']==[0] for op in operations))
        for extension in ('png','svg','pdf'):
            self.assertGreater((out/('bands_before_after.'+extension)).stat().st_size,1000)
        with np.load(out/'bands.npz') as data:
            before=data['before'] if 'before' in data else data['before_eV']
            after=data['after'] if 'after' in data else data['after_eV']
            np.testing.assert_allclose(after,before,atol=1e-10)

    def test_scalar(self):self.run_branch('scalar')
    def test_soc(self):self.run_branch('soc')
    def test_collinear(self):self.run_branch('collinear')
    def test_msg(self):self.run_branch('magnetic')
    def test_ssg(self):self.run_branch('ssg')

    def test_interleaved_soc_restores_hr_order(self):
        dest=self.base/'interleaved';shutil.copytree(self.demo/'soc',dest)
        h=entry.core.read_hr(dest/'wannier90_hr.dat')
        entry.core.write_hr(dest/'wannier90_hr.dat',entry.core.reorder(h,True))
        rows,centers,_=ai.projection_rows(dest/'wannier90.wout')
        order=[0,3,1,4,2,5]
        lines=['PROJECTIONS']+['| '+' '.join(map(str,rows[i]))+' |' for i in order]+['+---+']
        lines+=['WF centre and spread %d ( %s ) 0.'%(j+1,', '.join(map(str,centers[i]))) for j,i in enumerate(order)]
        (dest/'wannier90.wout').write_text('\n'.join(lines))
        args=self.args('soc',['--poscar',str(dest/'POSCAR'),'--hr',str(dest/'wannier90_hr.dat'),'--out',str(self.base/'interleaved output')])
        entry.execute(args)
        actual=entry.core.read_hr(args.out/'wannier90_hr_sym.dat')
        expected=entry.core.reorder(h,True)
        self.assertEqual(set(actual),set(expected))
        for key in actual:np.testing.assert_allclose(actual[key],expected[key],atol=1e-11)

    def test_poscar_coordinates_moments_and_selective_flags(self):
        path=self.base/'scaled POSCAR'
        path.write_text('test\n2\n1 0 0\n0 1 0\n0 0 1\nA B\n1 1\nSelective dynamics\nCartesian\n.5 0 0 T F T 1 2 3\n0 .5 0 F F F\n')
        lattice,atoms,moments,rows=ai.read_poscar(path)
        np.testing.assert_allclose(lattice,2*np.eye(3))
        np.testing.assert_allclose(atoms[0]['fractional'],[.5,0,0])
        np.testing.assert_allclose(moments,[[1,2,3],[0,0,0]])
        self.assertEqual(rows,[True,False])
        path.write_text(path.read_text().replace('\n2\n','\n-8\n'))
        np.testing.assert_allclose(ai.read_poscar(path)[0],lattice)
        path.write_text(path.read_text().replace('1 2 3','1 2'))
        with self.assertRaisesRegex(ValueError,'exactly three'):ai.read_poscar(path)

    def test_axes_require_explicit_source_convention(self):
        with self.assertRaisesRegex(ValueError,'Nontrivial local axes'):ai.axes([0,1,0],[0,0,1],'auto')
        x,z=ai.axes([0,1,0],[0,0,1],'transpose')
        np.testing.assert_allclose(x,[0,-1,0]);np.testing.assert_allclose(z,[0,0,1])

    def test_branch_and_missing_metadata_rejections(self):
        with self.assertRaisesRegex(ValueError,'order conflicts'):entry.prepare(self.args('ssg',['--ordering','collinear','--soc','yes']))
        with self.assertRaisesRegex(ValueError,'spinors disagrees'):entry.prepare(self.args('scalar',['--soc','yes']))
        args=self.args('scalar');dest=self.base/'bare';dest.mkdir()
        shutil.copy2(args.hr,dest/'bare_hr.dat');args.hr=dest/'bare_hr.dat'
        with self.assertRaisesRegex(ValueError,'Cannot infer orbitals'):entry.prepare(args)
        args.orbitals='1:s';args.home_cells='poscar';args.atomic_gauge=True
        cfg,_,_,_,provenance=entry.prepare(args)
        self.assertTrue(cfg['band_path'])
        self.assertIn('original reciprocal',provenance['band_path_source'])

    def test_explicit_ws_must_exist_and_source_conflicts_fail(self):
        with self.assertRaisesRegex(ValueError,'WS file does not exist'):entry.prepare(self.args('scalar',['--wsvec',str(self.base/'absent')]))
        path=self.base/'conflicting OUTCAR';path.write_text('vasp.6.test\nLSORBIT = T\nLNONCOLLINEAR = T\n')
        with self.assertRaisesRegex(ValueError,'LSORBIT conflicts'):entry.prepare(self.args('scalar',['--outcar',str(path)]))
        path.write_text('vasp.6.test\nISPIN = 2\nLSORBIT = F\nLNONCOLLINEAR = F\n')
        with self.assertRaisesRegex(ValueError,'ISPIN conflicts'):entry.prepare(self.args('scalar',['--outcar',str(path)]))

    def test_mismatched_paired_home_cells_rejected(self):
        dest=self.base/'bad pair';shutil.copytree(self.demo/'collinear',dest)
        path=dest/'wannier90.dn.wout';path.write_text(path.read_text().replace('( 0.0, 0.0, 0.0 )','( 4.0, 0.0, 0.0 )'))
        args=self.args('collinear',['--poscar',str(dest/'POSCAR'),'--up-hr',str(dest/'wannier90.up_hr.dat'),'--down-hr',str(dest/'wannier90.dn_hr.dat')])
        with self.assertRaisesRegex(ValueError,'bases or home cells differ'):entry.prepare(args)

if __name__=='__main__':unittest.main()
