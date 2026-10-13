"""Independent analytic regressions for paired collinear non-SOC HR projection.

These tests exercise physical output coefficients and spectra, rather than
accepting the implementation's own symmetry-residual evaluator as an oracle.
Run with: python -m unittest discover -s tests -p test_collinear.py -v
"""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'wannsym_core'))
import wannsym as w
import collinear as c


ZERO = (0, 0, 0)
TRICLINIC = np.array([[3.1, .31, .23], [0., 3.7, .41], [0., 0., 4.3]])


def scalar_basis(lattice, positions, symbols, projected=None):
    if projected is None:
        projected = list(range(len(positions)))
    count = len(projected)
    return w.Basis(np.array(lattice, float), np.array(positions, float),
                   list(symbols), list(projected), [['s'] for _ in projected],
                   np.tile(np.eye(3), (count, 1, 1)), False,
                   [1] * count, [slice(i, i + 1) for i in range(count)], count)


def tetragonal_basis():
    # The nonprojected oxygen removes the half translation that would otherwise
    # exchange the magnetic sites and enforce equal spectra at the same k.
    return scalar_basis(np.diag([3., 3., 5.]),
                        [[.5, 0., 0.], [0., .5, 0.], [0., 0., 0.]],
                        ['V', 'V', 'O'], projected=[0, 1])


def altermagnet(delta=2., tx=.4, ty=.1):
    pair = []
    for sign in (1, -1):
        h = {ZERO: np.diag([sign * delta, -sign * delta]).astype(complex)}
        for r in ((1, 0, 0), (-1, 0, 0)):
            h[r] = np.diag([tx, ty]).astype(complex)
        for r in ((0, 1, 0), (0, -1, 0)):
            h[r] = np.diag([ty, tx]).astype(complex)
        pair.append(h)
    return tuple(pair)


def assert_hr_equal(test, actual, expected, n, atol=2e-12):
    zero = np.zeros((n, n), complex)
    for r in set(actual) | set(expected):
        np.testing.assert_allclose(actual.get(r, zero), expected.get(r, zero),
                                   atol=atol, rtol=0,
                                   err_msg=f'Incorrect coefficient at R={r}')


def explicit_full_group_average(pair, basis, ops, real_gauge=False):
    """Literal full-group sum, independent of the production coset algorithm.

    Deliberately use simple, unbatched loops and no collinear private helpers.
    The complete operation list is used for each input channel, so choosing
    one exchanging representative cannot affect this reference calculation.
    """
    accumulated = ({}, {})
    for item in ops:
        for source_spin in (0, 1):
            destination_spin = source_spin if item.eta == 1 else 1 - source_spin
            destination = accumulated[destination_spin]
            for r, matrix in pair[source_spin].items():
                for i, source_rows in enumerate(basis.slices):
                    for j, source_columns in enumerate(basis.slices):
                        transformed_r = tuple(int(value) for value in
                            (item.op.rotation @ np.array(r) +
                             item.op.shifts[j] - item.op.shifts[i]))
                        if transformed_r not in destination:
                            destination[transformed_r] = np.zeros(
                                (basis.n, basis.n), dtype=complex)
                        target_rows = basis.slices[item.op.mapping[i]]
                        target_columns = basis.slices[item.op.mapping[j]]
                        block = (item.op.matrices[i] @ matrix[source_rows, source_columns]
                                 @ item.op.matrices[j].conj().T)
                        destination[transformed_r][target_rows, target_columns] += block / len(ops)
    result = []
    zero = np.zeros((basis.n, basis.n), dtype=complex)
    for channel in accumulated:
        all_r = set(channel) | {tuple(-np.array(r)) for r in channel}
        if not all_r:
            all_r = {ZERO}
        hermitian = {}
        for r in all_r:
            matrix = (channel.get(r, zero) +
                      channel.get(tuple(-np.array(r)), zero).conj().T) / 2
            hermitian[r] = matrix.real.astype(complex) if real_gauge else matrix
        result.append(hermitian)
    return tuple(result)


def noisy_p_orbital_fixture():
    """D4h geometry with unequal home cells, local frames, and complex noise."""
    basis = tetragonal_basis()
    basis.positions[0] += [1, -1, 1]
    basis.positions[1] += [-1, 2, -2]
    basis.orbitals = [['p'], ['p']]
    basis.sizes = [3, 3]
    basis.slices = [slice(0, 3), slice(3, 6)]
    basis.n = 6
    # Columns are orthonormal local x/y/z; site 2 has a rotated local frame.
    basis.axes[1] = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    rng = np.random.default_rng(20261011)
    supports = ([ZERO, (1, 2, 1), (-2, 1, 0)],
                [ZERO, (0, 1, -2), (1, 0, 0), (-1, 0, 0)])
    pair = tuple({r: rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
                  for r in support} for support in supports)
    return basis, pair


class CollinearTests(unittest.TestCase):
    def test_afm_preserves_altermagnetic_same_k_splitting(self):
        basis = tetragonal_basis()
        ops = c.select_operations(basis, [1., -1., 0.])
        self.assertEqual(sum(item.eta == 1 for item in ops), 8)
        self.assertEqual(sum(item.eta == -1 for item in ops), 8)
        initial = altermagnet()
        projected = c.joint_project(initial, basis, ops, real_gauge=True)
        for out, original in zip(projected, initial):
            assert_hr_equal(self, out, original, basis.n)
        kx = np.array([[.25, 0., 0.]])
        ky = np.array([[0., .25, 0.]])
        up_x, down_x = [w.bands(h, kx)[0] for h in projected]
        np.testing.assert_allclose(up_x, [-1.2, 2.2], atol=2e-12)
        np.testing.assert_allclose(down_x, [-1.8, 2.8], atol=2e-12)
        self.assertGreater(np.max(abs(up_x - down_x)), .59)
        # Exchanging spatial C4 acts on k as well as the spin label.
        np.testing.assert_allclose(up_x, w.bands(projected[1], ky)[0], atol=2e-12)

    def test_joint_onsite_average_and_idempotence(self):
        basis = tetragonal_basis()
        ops = c.select_operations(basis, [1., -1., 0.])
        initial = altermagnet()
        initial[0][ZERO][0, 0] += .4
        saved = deepcopy(initial)
        out = c.joint_project(initial, basis, ops, real_gauge=True)
        np.testing.assert_allclose(out[0][ZERO], np.diag([2.2, -2.]), atol=2e-12)
        np.testing.assert_allclose(out[1][ZERO], np.diag([-2., 2.2]), atol=2e-12)
        twice = c.joint_project(out, basis, ops, real_gauge=True)
        for spin in (0, 1):
            assert_hr_equal(self, twice[spin], out[spin], basis.n)
            assert_hr_equal(self, initial[spin], saved[spin], basis.n, atol=0.)
            zero = np.zeros((basis.n, basis.n), complex)
            for r, matrix in out[spin].items():
                opposite = tuple(-np.array(r))
                np.testing.assert_allclose(matrix,
                    out[spin].get(opposite, zero).conj().T, atol=2e-12)

    def test_ferromagnet_retains_exchange_splitting(self):
        basis = tetragonal_basis()
        ops = c.select_operations(basis, [1., 1., 0.])
        self.assertTrue(all(item.eta == 1 for item in ops))
        pair = ({ZERO: np.eye(2, dtype=complex) * 3.},
                {ZERO: np.eye(2, dtype=complex) * -2.})
        out = c.joint_project(pair, basis, ops, real_gauge=True)
        np.testing.assert_allclose(out[0][ZERO], np.eye(2) * 3., atol=2e-12)
        np.testing.assert_allclose(out[1][ZERO], np.eye(2) * -2., atol=2e-12)
        self.assertAlmostEqual(float(w.bands(out[0], np.zeros((1, 3)))[0, 0] -
                                     w.bands(out[1], np.zeros((1, 3)))[0, 0]), 5.)

    def test_complex_hopping_survives_unless_real_gauge_requested(self):
        basis = scalar_basis(TRICLINIC,
            [[.13, .21, .31], [.37, .11, .63], [.19, .73, .42]],
            ['V', 'O', 'Se'], projected=[0])
        ops = c.select_operations(basis, [1., 0., 0.])
        self.assertEqual(len(ops), 1)
        pair = ({ZERO: np.array([[1.]], complex),
                 (1, 0, 0): np.array([[.2j]]),
                 (-1, 0, 0): np.array([[-.2j]])},
                {ZERO: np.array([[-1.]], complex)})
        out = c.joint_project(pair, basis, ops, real_gauge=False)
        assert_hr_equal(self, out[0], pair[0], 1)
        self.assertAlmostEqual(w.bands(out[0], np.array([[.25, 0., 0.]]))[0, 0], .6)
        real_out = c.joint_project(pair, basis, ops, real_gauge=True)
        for matrix in real_out[0].values():
            np.testing.assert_allclose(matrix.imag, 0., atol=0.)
        self.assertAlmostEqual(w.bands(real_out[0], np.array([[.25, 0., 0.]]))[0, 0], 1.)
        # Imposing K never averages the exchange-split up and down on-site terms.
        self.assertAlmostEqual(real_out[1][ZERO][0, 0].real, -1.)

    def test_inversion_home_cell_shift_changes_offdiagonal_R(self):
        basis = scalar_basis(TRICLINIC, [[0., 0., 0.], [.5, 0., 0.]], ['V', 'O'])
        ops = c.select_operations(basis, [1., 2.])
        self.assertEqual(len(ops), 2)
        hopping = .6
        pair = ({ZERO: np.array([[0., hopping], [hopping, 0.]], complex)},
                {ZERO: np.zeros((2, 2), complex)})
        out = c.joint_project(pair, basis, ops)
        expected = {
            ZERO: np.array([[0., hopping / 2], [hopping / 2, 0.]], complex),
            (-1, 0, 0): np.array([[0., hopping / 2], [0., 0.]], complex),
            (1, 0, 0): np.array([[0., 0.], [hopping / 2, 0.]], complex),
        }
        assert_hr_equal(self, out[0], expected, 2)
        kpoints = np.array([[0., 0., 0.], [.25, 0., 0.], [.5, 0., 0.]])
        magnitudes = hopping * abs(np.cos(np.pi * kpoints[:, 0]))
        np.testing.assert_allclose(w.bands(out[0], kpoints),
                                   np.column_stack((-magnitudes, magnitudes)), atol=2e-12)
        repeat = c.joint_project(out, basis, ops)
        assert_hr_equal(self, repeat[0], out[0], 2)

    def test_weighted_asymmetric_support_is_zero_extended_once(self):
        basis = tetragonal_basis()
        ops = c.select_operations(basis, [1., -1., 0.])
        # Channel up contains x hopping on site A; channel down has no hopping.
        # Half the group keeps up/A/x and half sends it to down/B/y.  Thus each
        # surviving weighted coefficient is .4, not .8 and not weight-squared.
        physical_up = {ZERO: np.diag([2., -2.]).astype(complex),
                       (1, 0, 0): np.diag([.8, 0.]).astype(complex),
                       (-1, 0, 0): np.diag([.8, 0.]).astype(complex)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weighted_up_hr.dat'
            rows = [((1, 0, 0), 3), (ZERO, 1), ((-1, 0, 0), 2)]
            text = 'Weighted, nonlexicographic regression\n2\n3\n3 1 2\n'
            for r, weight in rows:
                for i, j in ((1, 1), (0, 1), (1, 0), (0, 0)):
                    z = physical_up[r][i, j] * weight
                    text += f'{r[0]} {r[1]} {r[2]} {i + 1} {j + 1} {z.real} {z.imag}\n'
            path.write_text(text)
            read_up = w.read_hr(path)
            assert_hr_equal(self, read_up, physical_up, 2)
            pair = (read_up, {ZERO: np.diag([-2., 2.]).astype(complex)})
            out = c.joint_project(pair, basis, ops)
            expected_up = {ZERO: np.diag([2., -2.]).astype(complex)}
            expected_down = {ZERO: np.diag([-2., 2.]).astype(complex)}
            for sign in (-1, 1):
                expected_up[(sign, 0, 0)] = np.diag([.4, 0.]).astype(complex)
                expected_down[(0, sign, 0)] = np.diag([0., .4]).astype(complex)
            assert_hr_equal(self, out[0], expected_up, 2)
            assert_hr_equal(self, out[1], expected_down, 2)
            for spin in (0, 1):
                target = Path(tmp) / f'roundtrip_{spin}_hr.dat'
                w.write_hr(target, out[spin])
                assert_hr_equal(self, w.read_hr(target), out[spin], 2)
            repeat = c.joint_project(out, basis, ops)
            for spin in (0, 1):
                assert_hr_equal(self, repeat[spin], out[spin], 2)

    def test_unprojected_moments_filter_the_group(self):
        basis = scalar_basis(TRICLINIC,
            [[0., 0., 0.], [.13, .21, .31], [-.13, -.21, -.31]],
            ['V', 'Fe', 'Fe'], projected=[0])
        structural = w.operations(basis, 1e-5)
        self.assertEqual(len(structural), 2)
        # Inversion preserves the projected V moment, but reverses the two
        # unprojected Fe moments. No common eta exists, so inversion is excluded.
        moments = np.array([1., 2., -2.])
        ops = c.select_operations(basis, moments)
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0].eta, 1)
        np.testing.assert_array_equal(moments[ops[0].full_mapping], moments)

    def test_coset_all_representatives_match_full_group_for_complex_hoppings(self):
        basis, pair = noisy_p_orbital_fixture()
        saved = deepcopy(pair)
        ops = c.select_operations(basis, [1., -1., 0.])
        self.assertEqual(len(ops), 16)
        exchanging = [index for index, item in enumerate(ops) if item.eta == -1]
        self.assertEqual(len(exchanging), 8)
        self.assertTrue(any(np.max(abs(item.op.shifts)) > 1 for item in ops))
        for real_gauge in (False, True):
            expected = explicit_full_group_average(pair, basis, ops, real_gauge)
            # The default representative must agree with the same full-group
            # average; the reference does not invoke joint_project at all.
            default = c.joint_project(pair, basis, ops, real_gauge=real_gauge)
            for spin in (0, 1):
                assert_hr_equal(self, default[spin], expected[spin], basis.n)
            if not real_gauge:
                self.assertGreater(max(np.max(abs(matrix.imag))
                    for channel in expected for matrix in channel.values()), 1e-6)
            for index in exchanging:
                with self.subTest(real_gauge=real_gauge, exchange_index=index):
                    actual = c.joint_project(pair, basis, ops,
                        real_gauge=real_gauge, exchange_index=index)
                    repeated = c.joint_project(actual, basis, ops,
                        real_gauge=real_gauge, exchange_index=index)
                    for spin in (0, 1):
                        assert_hr_equal(self, actual[spin], expected[spin], basis.n)
                        assert_hr_equal(self, repeated[spin], actual[spin], basis.n)
        for spin in (0, 1):
            assert_hr_equal(self, pair[spin], saved[spin], basis.n, atol=0.)

    def test_coset_ferromagnetic_subgroup_and_invalid_representatives(self):
        basis, pair = noisy_p_orbital_fixture()
        fm_ops = c.select_operations(basis, [1., 1., 0.])
        self.assertEqual(len(fm_ops), 16)
        self.assertTrue(all(item.eta == 1 for item in fm_ops))
        for real_gauge in (False, True):
            expected = explicit_full_group_average(pair, basis, fm_ops, real_gauge)
            actual = c.joint_project(pair, basis, fm_ops,
                                     real_gauge=real_gauge, exchange_index=None)
            repeated = c.joint_project(actual, basis, fm_ops,
                                       real_gauge=real_gauge, exchange_index=None)
            for spin in (0, 1):
                assert_hr_equal(self, actual[spin], expected[spin], basis.n)
                assert_hr_equal(self, repeated[spin], actual[spin], basis.n)
        with self.assertRaises(ValueError):
            c.joint_project(pair, basis, fm_ops, exchange_index=0)
        afm_ops = c.select_operations(basis, [1., -1., 0.])
        preserving = next(index for index, item in enumerate(afm_ops) if item.eta == 1)
        exchanging = next(index for index, item in enumerate(afm_ops) if item.eta == -1)
        for bad_index in (preserving, -1, len(afm_ops), float(exchanging),
                          str(exchanging), True, False):
            with self.subTest(invalid_exchange_index=repr(bad_index)):
                with self.assertRaises(ValueError):
                    c.joint_project(pair, basis, afm_ops, exchange_index=bad_index)

    def test_rejects_wrong_moments_spinor_and_matrix_dimensions(self):
        basis = tetragonal_basis()
        for moments in ([1., -1.], [0., 0., 0.], [1., np.nan, 0.]):
            with self.assertRaises(ValueError):
                c.select_operations(basis, moments)
        bad_basis = deepcopy(basis)
        bad_basis.spinor = True
        with self.assertRaises(ValueError):
            c.select_operations(bad_basis, [1., -1., 0.])
        ops = c.select_operations(basis, [1., -1., 0.])
        with self.assertRaises(ValueError):
            c.joint_project(({ZERO: np.eye(1)}, {ZERO: np.eye(2)}), basis, ops)
        with self.assertRaises(ValueError):
            c.joint_project(({ZERO: np.eye(2)}, {ZERO: np.eye(2)}), basis, [])
        with self.assertRaises(ValueError):
            c.joint_project(({(.5, 0, 0): np.eye(2)}, {ZERO: np.eye(2)}), basis, ops)


if __name__ == '__main__':
    unittest.main(verbosity=2)
