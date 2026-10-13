"""Non-SOC noncollinear spin-space operations and spinor HR projection.

Follows MOM2SSG findAllOp's atom/moment compatibility condition, solving
orthogonal spin maps directly for each spatial permutation (no SSG numbering).
Rank-two textures include both extensions, including the spin-only mirror.
Rank-one continuous stabilizers are deliberately outside this entrypoint.
"""
from dataclasses import dataclass
import numpy as np
import spglib
import wannsym as w
import magnetic as msg
from collinear import _full_mapping
from magnetic import project, residuals, bloch_residual, validate_hr, difference, action, hk, bloch_matrix

VERSION = '0.1.0'


@dataclass
class SpinSpaceOperation(msg.MagneticOperation):
    spin_rotation: np.ndarray
    moment_rank: int


def spin_maps(moments, targets, rank):
    """Orthogonal Procrustes; rank two has two possible normal extensions."""
    left, _, right = np.linalg.svd(moments.T @ targets)
    signs = (1., -1.) if rank == 2 else (1.,)
    return [(left @ np.diag([1., 1., sign]) @ right).T for sign in signs]


def select_operations(basis, moments, symprec=1e-5, moment_tol=1e-3, rank_tol=1e-6):
    w.require(basis.spinor, 'SSG requires a complete spinor basis')
    moments = np.asarray(moments, float)
    w.require(moments.shape == (len(basis.positions), 3) and np.isfinite(moments).all(),
              'Supply Cartesian vector moments for every structural atom')
    w.require(min(symprec, moment_tol, rank_tol) > 0, 'Invalid SSG tolerances')
    singular = np.linalg.svd(moments, compute_uv=False)
    rank = int(np.sum(singular > rank_tol * np.sqrt(len(moments))))
    w.require(rank in (2, 3), 'SSG runner requires a noncollinear rank-2 or rank-3 texture; rank-1 has a continuous spin stabilizer and needs the collinear workflow; rank-0 needs nonmagnetic treatment')
    species = {s: i+1 for i, s in enumerate(dict.fromkeys(basis.symbols))}
    sym = spglib.get_symmetry((basis.lattice.T, basis.positions, [species[s] for s in basis.symbols]), symprec=symprec)
    w.require(sym is not None, 'No structural operations')
    lookup = {site: i for i, site in enumerate(basis.atom_indices)}
    ops = []; tol = max(5*symprec, 1e-8)
    for rot, trans in zip(sym['rotations'], sym['translations']):
        full = _full_mapping(basis, rot, trans, tol)
        cart = basis.lattice @ rot @ np.linalg.inv(basis.lattice)
        w.require(np.allclose(cart.T @ cart, np.eye(3), atol=1e-7, rtol=0), 'Nonorthogonal spatial operation')
        for spin_rotation in spin_maps(moments, moments[full], rank):
            error = float(np.linalg.norm(moments[full] - moments @ spin_rotation.T, axis=1).max())
            if error > moment_tol:
                continue
            anti = bool(np.linalg.det(spin_rotation) < 0)
            # Improper action U on moments is represented as D(-U) T on spinors.
            spin = msg.spin_matrix(-spin_rotation if anti else spin_rotation)
            if anti: spin = spin @ msg.TSPIN
            mapping = []; shifts = []; matrices = []
            for i, site in enumerate(basis.atom_indices):
                target = full[site]
                w.require(target in lookup, 'Projected sites do not close under SSG')
                j = lookup[target]
                w.require(basis.orbitals[i] == basis.orbitals[j], 'Mapped orbital shells differ')
                local = basis.axes[j].T @ cart @ basis.axes[i]
                orbital = np.zeros((basis.sizes[i]//2,)*2, complex); start = 0
                for shell in basis.orbitals[i]:
                    size = w.DIMS[shell]
                    orbital[start:start+size, start:start+size] = w.get_any_rot_orb_twostep(shell, local)
                    start += size
                d = np.kron(orbital, spin)
                w.require(np.allclose(d.conj().T @ d, np.eye(len(d)), atol=1e-7, rtol=0), 'Orbital subspace is not closed/unitary')
                mapping.append(j)
                shifts.append(np.rint(rot @ basis.positions[site] + trans - basis.positions[target]).astype(int))
                matrices.append(d)
            ops.append(SpinSpaceOperation(w.Operation(rot, np.asarray(trans), mapping, np.asarray(shifts), matrices),
                                          anti, full, error, spin_rotation, rank))
    # For rank >=2, each spatial permutation has at most one spin map of each
    # parity. The shared validator also checks semilinear spinor composition.
    msg.validate_group(basis, ops, tol)
    for a in ops:
        for b in ops:
            rot = a.op.rotation @ b.op.rotation
            trans = a.op.rotation @ b.op.translation + a.op.translation
            matches = [c for c in ops if c.antiunitary == (a.antiunitary ^ b.antiunitary)
                       and np.array_equal(c.op.rotation, rot)
                       and np.linalg.norm(basis.lattice @ (trans-c.op.translation-np.rint(trans-c.op.translation))) < tol
                       and np.allclose(a.spin_rotation @ b.spin_rotation, c.spin_rotation, atol=2e-7, rtol=0)]
            w.require(len(matches) == 1, 'SSG spin maps are not closed; use documented symmetry-consistent moments rather than rounded/noisy values')
    return ops


def metadata(ops):
    records = msg.metadata(ops)
    for record, op in zip(records, ops):
        record.update(spin_rotation_cartesian=op.spin_rotation.tolist(), moment_rank=op.moment_rank,
                      spin_determinant=float(np.linalg.det(op.spin_rotation)))
    return records
