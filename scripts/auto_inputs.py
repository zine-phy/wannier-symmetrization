"""Construct audited case inputs from an extended POSCAR and Wannier metadata."""
from pathlib import Path
import re
import numpy as np

DIMS = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 't2g': 3}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def clean(line):
    return re.split(r'[#!]', line, maxsplit=1)[0].strip()


def number(value):
    return float(value.replace('D', 'E').replace('d', 'e'))


def read_poscar(path):
    lines = Path(path).read_text().splitlines()
    require(len(lines) >= 8, 'Truncated POSCAR')
    scale = np.array([number(x) for x in clean(lines[1]).split()])
    raw = np.array([[number(x) for x in clean(s).split()] for s in lines[2:5]])
    require(raw.shape == (3, 3) and np.isfinite(raw).all() and abs(np.linalg.det(raw)) > 1e-12, 'Invalid POSCAR lattice')
    if len(scale) == 1:
        require(scale[0] != 0, 'Zero POSCAR scale')
        factor = scale[0] if scale[0] > 0 else (-scale[0]/abs(np.linalg.det(raw)))**(1/3)
    else:
        require(len(scale) == 3 and np.all(scale > 0), 'POSCAR needs one scale or three positive Cartesian scales')
        factor = scale
    lattice = raw * factor
    symbols = clean(lines[5]).split()
    require(symbols and all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', s) for s in symbols), 'A VASP5 POSCAR with species names is required')
    counts = [int(x) for x in clean(lines[6]).split()]
    require(len(counts) == len(symbols) and min(counts) >= 0 and sum(counts) > 0, 'Invalid POSCAR species counts')
    i = 7; selective = clean(lines[i]).lower().startswith('s')
    if selective: i += 1
    mode = clean(lines[i]).lower(); i += 1
    require(mode.startswith(('d', 'c', 'k')), 'POSCAR coordinate mode must be Direct or Cartesian')
    atoms = []; moments = []; present = []
    for symbol, count in zip(symbols, counts):
        for _ in range(count):
            require(i < len(lines), 'Truncated POSCAR atom list')
            fields = clean(lines[i]).split(); i += 1
            require(len(fields) >= 3, 'Each atom needs three position coordinates')
            position = np.array([number(x) for x in fields[:3]])
            tail = fields[3:]
            if selective:
                require(len(tail) >= 3 and all(x.upper() in ('T', 'F') for x in tail[:3]), 'Selective dynamics requires three T/F flags before magnetic moments')
                tail = tail[3:]
            if tail and tail[-1] == symbol: tail = tail[:-1]
            require(len(tail) in (0, 3), 'Append exactly three Cartesian moment components, or none, after coordinates/Selective dynamics flags')
            moment = [number(x) for x in tail] if tail else [0., 0., 0.]
            if not mode.startswith('d'): position = (position * factor) @ np.linalg.inv(lattice)
            require(np.isfinite(position).all() and np.isfinite(moment).all(), 'Nonfinite atom coordinates or moments')
            atoms.append({'element': symbol, 'fractional': position.tolist()})
            moments.append(moment); present.append(bool(tail))
    return lattice, atoms, np.array(moments), present


def read_win(path):
    if path is None: return {}, {}
    settings = {}; blocks = {}; active = None
    for raw in Path(path).read_text().splitlines():
        line = clean(raw)
        if not line: continue
        match = re.match(r'begin\s+(\w+)', line, re.I)
        if match:
            active = match[1].lower(); require(active not in blocks, 'Duplicate WIN block: '+active); blocks[active] = []; continue
        if re.match(r'end\s+', line, re.I): active = None; continue
        if active: blocks[active].append(line)
        else:
            match = re.match(r'(\w+)\s*[=:]\s*(.*)', line)
            if match: settings[match[1].lower()] = match[2].strip()
    return settings, blocks


def truth(value):
    value = str(value).strip().lower().strip('.')
    require(value in ('true', 'false', 't', 'f'), 'Expected Boolean metadata, got '+value)
    return value in ('true', 't')


def check_win_structure(blocks, lattice):
    if 'unit_cell_cart' not in blocks: return
    rows = blocks['unit_cell_cart'][:]; factor = 1.
    if rows and len(rows[0].split()) == 1:
        unit = rows.pop(0).lower()
        require(unit in ('ang', 'angstrom', 'bohr'), 'Unsupported WIN lattice unit')
        if unit == 'bohr': factor = .529177210903
    values = np.array([[number(x) for x in line.split()] for line in rows]) * factor
    require(values.shape == (3, 3) and np.allclose(values, lattice, atol=1e-6, rtol=1e-6), 'POSCAR and WIN lattices differ; do not combine unrelated calculations')


def projection_rows(path):
    text = Path(path).read_text(); start = text.find('PROJECTIONS')
    require(start >= 0, 'WOUT has no PROJECTIONS table')
    records = []
    for line in text[start:].splitlines()[1:]:
        fields = line.strip().strip('|').split()
        if len(fields) == 13:
            try: values = [number(x) for x in fields]
            except ValueError: continue
            records.append(values)
        elif records and line.lstrip().startswith('+'): break
    require(records, 'No supported WOUT projection rows found')
    centers = re.findall(r'WF centre and spread\s+\d+\s*\(\s*([^)]*)\)', text)
    centers = [[number(x) for x in s.replace(',', ' ').split()] for s in centers]
    release = re.search(r'Release:\s*([^\n|]+)', text)
    return np.array(records), np.array(centers), release[1].strip() if release else 'unknown'


def map_atom(position, atoms, rounded=False):
    pos = np.array([a['fractional'] for a in atoms]); delta = pos-position; delta -= np.rint(delta)
    tol = .0051 if rounded else 1e-6
    hits = np.flatnonzero(np.max(abs(delta), axis=1) <= tol)
    require(len(hits) == 1, 'Projection center does not uniquely match POSCAR atoms; need higher-precision or explicit basis metadata')
    return int(hits[0])


def axes(x, z, convention):
    x = np.array(x, float); z = np.array(z, float)
    require(min(np.linalg.norm(x), np.linalg.norm(z)) > 1e-10, 'Zero local axis')
    x /= np.linalg.norm(x); z /= np.linalg.norm(z)
    require(abs(x@z) < 1e-7, 'Local axes are not orthogonal')
    a = np.column_stack((x, np.cross(z, x), z))
    if not np.allclose(a, np.eye(3), atol=1e-7):
        require(convention != 'auto', 'Nontrivial local axes: specify --axes-convention physical or transpose after auditing the source interface')
        if convention == 'transpose': a = a.T
    return a[:, 0].tolist(), a[:, 2].tolist()


def basis_from_wout(path, atoms, lattice, dimension, spinor, spin_order, convention, home_cells):
    rows, centers, release = projection_rows(path)
    require(len(rows) == dimension, 'WOUT projection count differs from HR dimension (unsupported selected/random projections or mismatched files)')
    order = 'scalar'; center_indices = list(range(len(rows)))
    if spinor:
        require(len(rows) % 2 == 0, 'Odd spinor dimension')
        adjacent = np.allclose(rows[::2], rows[1::2], atol=1e-8)
        halves = np.allclose(rows[:len(rows)//2], rows[len(rows)//2:], atol=1e-8)
        if spin_order == 'auto':
            require(adjacent or halves, 'Cannot infer spin order from WOUT; explicit metadata is required')
            require(not(adjacent and halves) or len(rows) == 2, 'Ambiguous spin ordering in WOUT')
            order = 'interleaved' if adjacent else 'all-up-then-all-down'
        else: order = spin_order
        require((order == 'interleaved' and adjacent) or (order == 'all-up-then-all-down' and halves), 'Declared spin order conflicts with projection table')
        center_indices = list(range(0, dimension, 2)) if order == 'interleaved' else list(range(dimension//2))
        rows = rows[center_indices]
    sites = []; row_atoms = []; i = 0
    while i < len(rows):
        row = rows[i]; atom = map_atom(row[:3], atoms, rounded=True)
        ell = int(row[3]); require(ell in range(4), 'Automatic WOUT parsing supports complete s/p/d/f shells only')
        shell = 'spdf'[ell]; count = DIMS[shell]
        require(i+count <= len(rows), 'Truncated orbital shell')
        for mr, q in enumerate(rows[i:i+count], 1):
            require(np.allclose(q[:3], row[:3]) and q[3] == ell and q[4] == mr and q[5] == row[5]
                    and np.allclose(q[6:12], row[6:12]), 'Orbital order is not a complete real-harmonic shell; supply audited basis JSON')
        x, z = axes(row[9:12], row[6:9], convention)
        if sites and sites[-1]['atom'] == atom+1:
            require(sites[-1]['x'] == x and sites[-1]['z'] == z and shell not in sites[-1]['shells'], 'Repeated radial shells or different axes on one site need a richer representation')
            sites[-1]['shells'].append(shell)
        else:
            require(all(site['atom'] != atom+1 for site in sites), 'Interleaved site blocks need an explicit HR permutation')
            sites.append({'atom': atom+1, 'shells': [shell], 'x': x, 'z': z})
        row_atoms.extend([atom]*count); i += count
    home_note = 'Explicit POSCAR representatives retained'
    if home_cells == 'auto':
        require(len(centers) >= dimension, 'WOUT lacks final WF centers: specify --home-cells poscar only after verifying HR home cells')
        cf = centers[-dimension:] @ np.linalg.inv(lattice)
        spatial_cf = cf[center_indices]
        all_shifts = {}
        for n, atom in enumerate(row_atoms):
            shift = np.rint(spatial_cf[n]-np.array(atoms[atom]['fractional'])).astype(int)
            if spinor:
                partner = center_indices[n]+1 if order == 'interleaved' else center_indices[n]+dimension//2
                other = np.rint(cf[partner]-np.array(atoms[atom]['fractional'])).astype(int)
                require(np.array_equal(shift, other), 'Spin partners have inconsistent home cells')
            require(atom not in all_shifts or np.array_equal(all_shifts[atom], shift), 'WFs on one site have inconsistent home cells; inspect the Wannier gauge')
            all_shifts[atom] = shift
        for atom, shift in all_shifts.items(): atoms[atom]['fractional'] = (np.array(atoms[atom]['fractional'])+shift).tolist()
        home_note = 'Integer home shifts inferred consistently from final WF centers; intra-cell displacements not used as atom positions'
    return sites, order, release, home_note


def basis_from_win(blocks, atoms, spinor, spin_order, convention):
    sites = []
    for line in blocks.get('projections', []):
        parts = [s.strip() for s in line.split(':')]
        require(len(parts) >= 2, 'Unsupported projection syntax: '+line)
        selector = parts[0]
        if re.match(r'f\s*=', selector, re.I):
            pos = [number(v) for v in selector.split('=', 1)[1].replace(',', ' ').split()]
            require(len(pos) == 3, 'Projection f= needs three fractional coordinates')
            ids = [map_atom(np.array(pos), atoms)]
        else:
            ids = [i for i, atom in enumerate(atoms) if atom['element'].lower() == selector.lower()]
            require(ids, 'Unsupported projection selector or absent species: '+selector)
        shells = []; x = [1, 0, 0]; z = [0, 0, 1]
        for part in parts[1:]:
            match = re.fullmatch(r'([xz])\s*=\s*(.*)', part, re.I)
            if match:
                vec = [number(v) for v in match[2].replace(',', ' ').split()]
                require(len(vec) == 3, 'Projection axis needs three components')
                if match[1].lower() == 'x': x = vec
                else: z = vec
                continue
            for shell in part.replace(';', ',').split(','):
                shell = shell.strip().lower()
                if re.fullmatch(r'l\s*=\s*[0-3]', shell): shell = 'spdf'[int(shell.split('=')[1])]
                require(shell in DIMS, 'Unsupported projection syntax; use WOUT or explicit --orbitals: '+shell)
                shells.append(shell)
        require(shells, 'Empty projection shell list'); x, z = axes(x, z, convention)
        for atom in ids:
            require(all(site['atom'] != atom+1 for site in sites), 'Use a single combined projection entry per site')
            sites.append({'atom': atom+1, 'shells': shells[:], 'x': x, 'z': z})
    require(sites, 'Cannot infer orbitals from POSCAR/HR alone. Provide matching WIN/WOUT, --orbitals, or --basis-json')
    require(not spinor or spin_order != 'auto', 'WIN alone does not certify spin ordering; provide --spin-order')
    return sites, spin_order if spinor else 'scalar'


def path_from_metadata(blocks, lattice, atoms, magnetic, points, symprec):
    if blocks.get('kpoint_path'):
        result = []
        for line in blocks['kpoint_path']:
            fields = line.split(); require(len(fields) == 8, 'Invalid WIN kpoint_path segment')
            result.append({'start': [number(x) for x in fields[1:4]], 'end': [number(x) for x in fields[5:8]],
                           'labels': [fields[0].replace('GAMMA', 'Gamma'), fields[4].replace('GAMMA', 'Gamma')], 'points': points})
        return result, 'Matching WIN kpoint_path'
    import seekpath
    types = {s: i+1 for i, s in enumerate(dict.fromkeys(a['element'] for a in atoms))}
    data = seekpath.get_path_orig_cell((lattice, [a['fractional'] for a in atoms], [types[a['element']] for a in atoms]),
                                      with_time_reversal=not magnetic, symprec=symprec)
    result = [{'start': data['point_coords'][a], 'end': data['point_coords'][b],
               'labels': [a.replace('GAMMA', 'Gamma'), b.replace('GAMMA', 'Gamma')], 'points': points} for a, b in data['path']]
    return result, 'SeeK-path chemical-structure path in original reciprocal basis; primitive-cell BZ path if input is a supercell; not a magnetic little-group labeling'
