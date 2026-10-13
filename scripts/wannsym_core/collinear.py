"""Collinear, non-SOC symmetry projection of separate up/down HR models.

Spatial operations act on real atomic orbitals.  An operation is retained when
it preserves the signed moment pattern or reverses the entire pattern; the
latter is accompanied by an independent spin rotation that exchanges channels.
This is not the axial-vector magnetic-space-group rule used with SOC.

The finite HR inputs are zero-extended, rather than extrapolating missing
hoppings.  First average the channel-preserving subgroup, then average that
result with one channel-exchanging image.  This coset construction equals the
complete retained-group projection.  No input is modified, no files are written,
and complex hoppings are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import spglib

from wannsym import Basis, Operation, DIMS, get_any_rot_orb_twostep, require

VERSION = '1.1.0'


@dataclass
class TaggedOperation:
    op: Operation
    eta: int
    full_mapping: list[int]


def _full_mapping(basis, rotation, translation, tolerance):
    mapping = []
    for site, position in enumerate(basis.positions):
        transformed = rotation @ position + translation
        candidates = []
        for target, target_position in enumerate(basis.positions):
            if basis.symbols[target] != basis.symbols[site]:
                continue
            delta = transformed - target_position
            if np.linalg.norm(basis.lattice @ (delta - np.rint(delta))) < tolerance:
                candidates.append(target)
        require(len(candidates) == 1,
                f'Ambiguous or unmatched structural site mapping for atom {site + 1}')
        mapping.append(candidates[0])
    require(len(set(mapping)) == len(mapping), 'Full atom mapping is not a permutation')
    return mapping


def _validate_group(basis, tagged, tolerance):
    """Check closure of the finite space-group quotient and the spin labels."""
    buckets = {}
    for index, item in enumerate(tagged):
        key = tuple(item.op.rotation.ravel())
        buckets.setdefault(key, []).append(index)
    identity_found = False
    for item in tagged:
        op = item.op
        if (np.array_equal(op.rotation, np.eye(3, dtype=int)) and
                np.linalg.norm(basis.lattice @ (op.translation -
                                                np.rint(op.translation))) < tolerance):
            require(item.eta == 1, 'The identity must preserve the spin channel')
            identity_found = True
    require(identity_found, 'The selected group has no identity')

    for left in tagged:
        for right in tagged:
            rotation = left.op.rotation @ right.op.rotation
            translation = left.op.rotation @ right.op.translation + left.op.translation
            candidates = []
            for index in buckets.get(tuple(rotation.ravel()), []):
                delta = translation - tagged[index].op.translation
                if np.linalg.norm(basis.lattice @ (delta - np.rint(delta))) < tolerance:
                    candidates.append(index)
            require(len(candidates) == 1,
                    'Selected operations are not a unique closed group; check moment/symmetry tolerances')
            product = tagged[candidates[0]]
            require(product.eta == left.eta * right.eta,
                    'Spin-channel labels do not close; the moment tolerance is inconsistent')
            composition = [left.full_mapping[j] for j in right.full_mapping]
            require(composition == product.full_mapping, 'Full atom maps fail group composition')
            for i, intermediate in enumerate(right.op.mapping):
                require(left.op.mapping[intermediate] == product.op.mapping[i],
                        'Projected atom maps fail group composition')
                composed = left.op.matrices[intermediate] @ right.op.matrices[i]
                require(np.allclose(composed, product.op.matrices[i], atol=1e-7, rtol=0),
                        'Orbital representations fail group composition')
            shifts = (right.op.shifts @ left.op.rotation.T +
                      left.op.shifts[right.op.mapping] - product.op.shifts)
            require(np.all(shifts == shifts[0]),
                    'Site home-cell shifts fail group composition')


def select_operations(basis: Basis, moments, symprec=1e-5, moment_tol=1e-3):
    """Select and validate the channel-preserving/exchanging symmetry group.

    ``moments`` is one signed scalar per *structural* atom, including atoms not
    present in the Wannier basis.  Signs refer to one common spin axis and must
    correspond to the labels on the two HR files.  Magnitudes participate in
    selection.  ``symprec`` is in Angstrom and ``moment_tol`` in the units of the
    supplied moments (normally Bohr magnetons).

    An accepted operation obeys m[map(a)] = eta*m[a].  Local orbital axes and
    unwrapped site home cells are read from ``basis``.  Only retained operations
    are required to close the projected orbital basis.
    """
    require(not basis.spinor, 'Collinear non-SOC projection requires a scalar orbital basis')
    require(symprec > 0 and moment_tol > 0, 'Tolerances must be positive')
    moments = np.asarray(moments, dtype=float)
    require(moments.shape == (len(basis.positions),) and np.isfinite(moments).all(),
            'Provide one finite signed moment for every structural atom')
    require(np.max(np.abs(moments)) > moment_tol,
            'All moments are zero within tolerance; use the nonmagnetic workflow')
    species = {symbol: i + 1 for i, symbol in enumerate(dict.fromkeys(basis.symbols))}
    cell = (basis.lattice.T, basis.positions, [species[s] for s in basis.symbols])
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    require(symmetry is not None, 'spglib could not find structural symmetry')
    atom_lookup = {site: i for i, site in enumerate(basis.atom_indices)}
    inverse_lattice = np.linalg.inv(basis.lattice)
    tolerance = max(5 * symprec, 1e-8)
    selected = []
    for rotation, translation in zip(symmetry['rotations'], symmetry['translations']):
        rotation = np.asarray(rotation, dtype=int)
        translation = np.asarray(translation, dtype=float)
        full_mapping = _full_mapping(basis, rotation, translation, tolerance)
        errors = {eta: float(np.max(np.abs(moments[full_mapping] - eta * moments)))
                  for eta in (1, -1)}
        labels = [eta for eta, error in errors.items() if error <= moment_tol]
        require(len(labels) <= 1, 'Ambiguous preserving/exchanging moment assignment')
        if not labels:
            continue
        eta = labels[0]
        cart = basis.lattice @ rotation @ inverse_lattice
        require(np.allclose(cart.T @ cart, np.eye(3), atol=1e-7, rtol=0),
                'Nonorthogonal spatial operation')
        mapping, shifts, matrices = [], [], []
        for i, site in enumerate(basis.atom_indices):
            target = full_mapping[site]
            require(target in atom_lookup,
                    'Projected atoms do not close under the selected magnetic symmetry')
            j = atom_lookup[target]
            require(basis.orbitals[i] == basis.orbitals[j],
                    'Magnetically mapped sites have incompatible orbital shells')
            mapping.append(j)
            delta = rotation @ basis.positions[site] + translation - basis.positions[target]
            shifts.append(np.rint(delta).astype(int))
            local = basis.axes[j].T @ cart @ basis.axes[i]
            dim = sum(DIMS[orb] for orb in basis.orbitals[i])
            orbital = np.zeros((dim, dim), dtype=complex)
            start = 0
            for orb in basis.orbitals[i]:
                size = DIMS[orb]
                orbital[start:start + size, start:start + size] = get_any_rot_orb_twostep(orb, local)
                start += size
            require(np.allclose(orbital.conj().T @ orbital, np.eye(dim), atol=1e-7, rtol=0),
                    'Projected orbital subspace is not closed or unitary')
            matrices.append(orbital)
        require(len(set(mapping)) == len(mapping), 'Projected site mapping is not a permutation')
        selected.append(TaggedOperation(Operation(rotation, translation, mapping,
                                                 np.asarray(shifts), matrices), eta, full_mapping))
    require(bool(selected), 'No magnetic symmetry operations were selected')
    _validate_group(basis, selected, tolerance)
    return selected


def _validate_pair(pair, basis):
    require(not basis.spinor, 'Separate collinear channels need a scalar basis')
    require(len(pair) == 2, 'Supply exactly two HR dictionaries: (up, down)')
    result = []
    for channel in pair:
        require(bool(channel), 'Empty HR channel')
        clean = {}
        for key, matrix in channel.items():
            r = np.asarray(key)
            require(r.shape == (3,) and np.isfinite(r).all() and
                    np.array_equal(r, np.rint(r)), 'Invalid integer HR lattice vector')
            r = tuple(int(v) for v in r)
            matrix = np.asarray(matrix, dtype=complex)
            require(matrix.shape == (basis.n, basis.n) and np.isfinite(matrix).all(),
                    'HR matrix dimensions or finite values disagree with the scalar basis')
            require(r not in clean, 'Duplicate normalized HR lattice vector')
            clean[r] = matrix
        result.append(clean)
    return tuple(result)


def _hermitize(channel, n):
    support = set(channel) | {tuple(-np.asarray(r)) for r in channel}
    if not support:
        support = {(0, 0, 0)}
    zero = np.zeros((n, n), dtype=complex)
    return {r: (channel.get(r, zero) +
                channel.get(tuple(-np.asarray(r)), zero).conj().T) / 2
            for r in sorted(support)}


def _projection_choice(ops, exchange_index):
    require(bool(ops), 'A nonempty closed group is required')
    require(all(item.eta in (-1, 1) for item in ops), 'Invalid spin-channel label')
    preserving = [i for i, item in enumerate(ops) if item.eta == 1]
    exchanging = [i for i, item in enumerate(ops) if item.eta == -1]
    require(bool(preserving), 'The group must contain a channel-preserving subgroup')
    require(len(exchanging) in (0, len(preserving)),
            'Channel-exchanging operations must form an index-two coset')
    if exchange_index is None:
        selected = exchanging[0] if exchanging else None
    else:
        require(isinstance(exchange_index, (int, np.integer)) and
                not isinstance(exchange_index, (bool, np.bool_)),
                'exchange_index must be a zero-based integer index in the full operation list')
        selected = int(exchange_index)
        require(0 <= selected < len(ops), 'exchange_index is outside the full operation list')
        require(ops[selected].eta == -1, 'exchange_index must select a channel-exchanging operation')
    return preserving, exchanging, selected


def _average_actions(pair, basis, ops):
    """Average selected forward actions without Hermitian or K projection."""
    output = ({}, {})
    scale = 1.0 / len(ops)
    for spin, channel in enumerate(pair):
        keys = sorted(channel)
        vectors = np.asarray(keys, dtype=int)
        matrices = np.asarray([channel[r] for r in keys])
        for item in ops:
            op = item.op
            target = output[spin if item.eta == 1 else 1 - spin]
            rotated_vectors = vectors @ op.rotation.T
            for i, si in enumerate(basis.slices):
                ti = basis.slices[op.mapping[i]]
                for j, sj in enumerate(basis.slices):
                    blocks = matrices[:, si, sj]
                    nonzero = np.any(blocks != 0, axis=(1, 2))
                    if not np.any(nonzero):
                        continue
                    tj = basis.slices[op.mapping[j]]
                    transformed = ((op.matrices[i] @ blocks[nonzero]) @
                                   op.matrices[j].conj().T) * scale
                    mapped_vectors = (rotated_vectors[nonzero] + op.shifts[j] - op.shifts[i])
                    for r, block in zip(mapped_vectors, transformed):
                        key = tuple(int(v) for v in r)
                        if key not in target:
                            target[key] = np.zeros((basis.n, basis.n), dtype=complex)
                        target[key][ti, tj] += block
    return tuple(channel if channel else {(0, 0, 0): np.zeros((basis.n, basis.n), complex)}
                 for channel in output)


def joint_project(pair, basis: Basis, ops, real_gauge=False, exchange_index=None):
    """Return new (up, down) HRs using a subgroup plus one exchange operation.

    ``ops`` is the complete closed group returned by ``select_operations``.
    Let H be its channel-preserving subgroup.  First form P_H(input), then,
    when exchanging operations exist, return (I + U_a) P_H(input) / 2 before
    Hermitian/K projection.  ``exchange_index`` chooses a in the *full*,
    zero-based operation list; None chooses the first exchanging operation.
    For a group with no exchange, only the preserving-subgroup average is used.

    H is normal of index two, a^2 belongs to H, and G = H union aH.  Therefore
    any exchanging representative gives exactly the full Reynolds projection.
    Averaging just input and U_a(input) without P_H is generally insufficient.

    A forward action maps a block to
      (eta*s, g*R + L_j - L_i, map(i), map(j))
    with value D_i H_s(R)[i,j] D_j^dagger.  Missing coefficients are zero.
    Hermitian projection uses the union of R and -R, even when input support
    differs between channels.  Optional ``real_gauge`` additionally imposes
    channel-wise K: H_s(R)=H_s(R)*; this is *not* physical spin time reversal.
    The caller must establish that both inputs use this real orbital gauge and
    share the same orbital labels, home cells, axes, and cross-channel phases.
    """
    pair = _validate_pair(pair, basis)
    preserving, _, selected = _projection_choice(ops, exchange_index)
    if real_gauge:
        require(all(np.max(np.abs(d.imag)) < 1e-10 for item in ops for d in item.op.matrices),
                'Channel-wise K does not commute with a complex orbital representation')
    output = _average_actions(pair, basis, [ops[index] for index in preserving])
    if selected is not None:
        exchanged = _average_actions(output, basis, [ops[selected]])
        zero = np.zeros((basis.n, basis.n), dtype=complex)
        output = tuple({r: (original.get(r, zero) + image.get(r, zero)) / 2
                        for r in sorted(set(original) | set(image))}
                       for original, image in zip(output, exchanged))
    output = tuple(_hermitize(channel, basis.n) for channel in output)
    if real_gauge:
        output = tuple({r: matrix.real.astype(complex) for r, matrix in channel.items()}
                       for channel in output)
    return output


def residuals(pair, basis: Basis, ops):
    """Return maximum-entry errors in eV; K is diagnostic unless requested.

    ``real_gauge_K_max_eV`` is never a test of physical spin time reversal.
    A nonzero value is allowed when projection used ``real_gauge=False``.
    """
    pair = _validate_pair(pair, basis)
    require(bool(ops), 'A nonempty symmetry group is required')
    zero = np.zeros((basis.n, basis.n), dtype=complex)
    herm = [0.0, 0.0]
    conjugation = 0.0
    preserving = exchanging = 0.0
    keys = sorted(set(pair[0]) | set(pair[1]))
    vectors = np.asarray(keys, dtype=int)
    for spin, channel in enumerate(pair):
        for r, matrix in channel.items():
            herm[spin] = max(herm[spin], float(np.max(np.abs(
                matrix - channel.get(tuple(-np.asarray(r)), zero).conj().T))))
            conjugation = max(conjugation, float(np.max(np.abs(matrix - matrix.conj()))))
        matrices = np.asarray([channel.get(r, zero) for r in keys])
        for item in ops:
            op = item.op
            target = pair[spin if item.eta == 1 else 1 - spin]
            rotated_vectors = vectors @ op.rotation.T
            for i, si in enumerate(basis.slices):
                ti = basis.slices[op.mapping[i]]
                for j, sj in enumerate(basis.slices):
                    tj = basis.slices[op.mapping[j]]
                    transformed = ((op.matrices[i] @ matrices[:, si, sj]) @
                                   op.matrices[j].conj().T)
                    mapped_vectors = rotated_vectors + op.shifts[j] - op.shifts[i]
                    actual = np.asarray([target.get(tuple(int(v) for v in r), zero)[ti, tj]
                                         for r in mapped_vectors])
                    error = float(np.max(np.abs(transformed - actual)))
                    if item.eta == 1:
                        preserving = max(preserving, error)
                    else:
                        exchanging = max(exchanging, error)
    return {'hermiticity_max_eV': max(herm),
            'hermiticity_up_max_eV': herm[0], 'hermiticity_down_max_eV': herm[1],
            'channel_preserving_max_eV': preserving,
            'channel_exchanging_max_eV': exchanging,
            'magnetic_group_max_eV': max(preserving, exchanging),
            'real_gauge_K_max_eV': conjugation}


def operation_metadata(ops):
    """Return a JSON-compatible audit trail in the operation order used."""
    return [{'index': index, 'eta': item.eta,
             'kind': 'preserve' if item.eta == 1 else 'exchange',
             'rotation': item.op.rotation.tolist(),
             'translation': item.op.translation.tolist(),
             'full_mapping_zero_based': list(item.full_mapping),
             'projected_mapping_zero_based': list(item.op.mapping),
             'home_cell_shifts': item.op.shifts.tolist(),
             'orbital_matrices_real': [d.real.tolist() for d in item.op.matrices],
             'orbital_matrices_imag': [d.imag.tolist() for d in item.op.matrices]}
            for index, item in enumerate(ops)]


def projection_metadata(ops, exchange_index=None):
    """Describe the coset projection and its selected full-list operation."""
    preserving, exchanging, selected = _projection_choice(ops, exchange_index)
    if selected is None:
        selection = 'no-exchanging-operations'
    else:
        selection = 'first-exchanging-operation' if exchange_index is None else 'explicit-index'
    return {'method': 'preserving-subgroup-average-then-one-exchange',
            'formula': 'P_G = (I + U_a) P_H / 2' if selected is not None else 'P_G = P_H',
            'full_group_order': len(ops),
            'preserving_subgroup_order': len(preserving),
            'exchanging_coset_size': len(exchanging),
            'preserving_operation_indices': preserving,
            'selected_exchange_index': selected,
            'selection': selection,
            'selected_exchange_operation': (operation_metadata(ops)[selected]
                                            if selected is not None else None)}
