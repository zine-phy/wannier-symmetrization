"""End-to-end relocation checks, including mandatory band artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class PortableCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='wannsym relocation ')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        cls.bundle = cls.base / 'copied skill with spaces'
        shutil.copytree(Path(__file__).resolve().parents[1], cls.bundle,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', 'demo', 'runs'))
        cls.work = cls.base / 'unrelated working directory'; cls.work.mkdir()
        cls.demo = cls.work / 'synthetic inputs'
        cls.env = dict(os.environ, PYTHONPATH='', WANNSYM_FONT='DejaVu Serif',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        result = cls.execute([cls.bundle / 'examples' / 'make_demo.py', '--out', cls.demo])
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def execute(cls, arguments):
        return subprocess.run([sys.executable, *map(str, arguments)], cwd=cls.work,
                              env=cls.env, capture_output=True, text=True, timeout=120)

    def run_branch(self, branch):
        out = self.work / (branch + ' result')
        args = [self.bundle / 'scripts' / ('symmetrize_collinear_hr.py' if branch == 'collinear'
                else 'symmetrize_ssg_hr.py' if branch == 'ssg' else 'symmetrize_magnetic_hr.py' if branch == 'magnetic' else 'symmetrize_hr.py')]
        if branch == 'collinear':
            args += ['--up-hr', self.demo / 'up_hr.dat', '--down-hr', self.demo / 'down_hr.dat']
        else:
            args += ['--hr', self.demo / f'{branch}_hr.dat']
        args += ['--case-json', self.demo / f'{branch}.json', '--ws-mode', 'raw', '--out', out]
        result = self.execute(args)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads((out / 'manifest.json').read_text())['status'], 'completed')
        self.assertTrue((out / 'validation.json').is_file())
        for ext in ['png', 'pdf', 'svg']:
            self.assertGreater((out / f'bands_before_after.{ext}').stat().st_size, 1000)
        with np.load(out / 'bands.npz') as spectra:
            before = spectra['before'] if 'before' in spectra else spectra['before_eV']
            after = spectra['after'] if 'after' in spectra else spectra['after_eV']
            # Each analytic fixture is already invariant under its intended group.
            np.testing.assert_allclose(after, before, atol=1e-11)
            if branch == 'collinear':
                np.testing.assert_allclose(after[0] - after[1], 1.4, atol=1e-11)
        if branch == 'soc':
            validation = json.loads((out / 'validation.json').read_text())
            self.assertLess(validation['eight_TRIM_Kramers_max_eV'], 1e-11)
        if branch == 'magnetic':
            validation = json.loads((out / 'validation.json').read_text())
            self.assertGreater(validation['after']['pure_TR_difference_eV'], 1.)

    def test_ssg_relocation(self): self.run_branch('ssg')

    def test_scalar_relocation(self): self.run_branch('scalar')
    def test_soc_relocation(self): self.run_branch('soc')
    def test_collinear_relocation(self): self.run_branch('collinear')
    def test_magnetic_relocation(self): self.run_branch('magnetic')

    def test_final_run_requires_band_path(self):
        for branch in ['scalar', 'collinear', 'magnetic', 'ssg']:
            with self.subTest(branch=branch):
                cfg = json.loads((self.demo / f'{branch}.json').read_text()); cfg.pop('band_path')
                config = self.work / f'{branch}_missing_path.json'; config.write_text(json.dumps(cfg))
                script = 'symmetrize_collinear_hr.py' if branch == 'collinear' else (
                    'symmetrize_ssg_hr.py' if branch == 'ssg' else 'symmetrize_magnetic_hr.py' if branch == 'magnetic' else 'symmetrize_hr.py')
                args = [self.bundle / 'scripts' / script, '--case-json', config, '--ws-mode', 'raw',
                        '--out', self.work / f'{branch}_must_not_complete']
                if branch == 'collinear':
                    args += ['--up-hr', self.demo / 'up_hr.dat', '--down-hr', self.demo / 'down_hr.dat']
                else:
                    args += ['--hr', self.demo / f'{branch}_hr.dat']
                result = self.execute(args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('band', result.stderr + result.stdout)


if __name__ == '__main__': unittest.main()
