# Code provenance and license

This folder contains a maintained Python 3 implementation based on orbital/spin rotation helpers from [WannierTools wannhr_symm](https://github.com/quanshengwu/wannier_tools/tree/master/utility/wannhr_symm), with new nonmagnetic, paired-collinear and magnetic-spinor projectors and workflow runners. It is not an official WannierTools release and is distinct from [ccao/WannSymm](https://github.com/ccao/WannSymm).

Upstream rotation-helper authorship is attributed to Changming Yue; preserved module headers include the original attributions, including the Euler-angle routine's attribution to G. Winker. The distribution carries the upstream GNU GPL version 3 license in [LICENSE](../LICENSE). Retain this license and source attribution when redistributing.

The inspected upstream helper fingerprints (before the Python 3 port) are:

| Relative upstream file under wannhr_symm/lib | SHA-256 |
| --- | --- |
| get_euler_angle.py | ff2a06ad78503f736b358611f07c6392c3e31227d904bc6ee64c851eee55046b |
| get_orb_rotmat_twostep.py | cb93ed149e046cc5e9abdc08e42b980af5442c3971d5ef9439516df353a094e4 |
| rotate.py | 2fb3a2424ddc84083905fe333bfa73fc7c2d1de5f6afbe2d453ddc2ed71ae94d |

The maintained implementation corrects Python 3 compatibility, weighted R handling and complex-block retention, and adds explicit gauge audits, operation validation, separate-channel subgroup/coset projection, semilinear MSG projection and mandatory band comparisons. No proprietary VASP source or private calculation inputs are included. Core and runner versions are recorded at runtime; code hashes identify the actual executed files.

Related public resources: [magnetic utility and examples](https://github.com/quanshengwu/wannier_tools/tree/master/utility/wannhr_symm_Mag), [upstream symmetrization notes](https://github.com/quanshengwu/wannier_tools/blob/master/doc/source/symmetrization.rst). Public links identify sources; they are not runtime dependencies.

## Noncollinear NSOC operation search

The SSG finder follows the operation compatibility logic of [MOM2SSG](https://github.com/zine-phy/MOM2SSG), inspected at commit `2b67cc45d66f81d26ae9af6453e3652b7791f06d`. It independently solves spin maps per spatial atom permutation instead of importing upstream classification code or point-group enumeration. The coplanar representative convention was checked against `findAllOp` and `twoD2threeD`; full symmetrization includes the spin-only mirror as documented in [SSG](ssg.md). MOM2SSG is MIT-licensed; its notice is retained in [licenses/MOM2SSG-MIT.txt](../licenses/MOM2SSG-MIT.txt). No classification database or compiled library is bundled.

## Automatically generated comparison paths

[SeeK-path](https://seekpath.readthedocs.io/en/latest/maindoc.html) generates fallback paths via `get_path_orig_cell`, without rotating the HR basis. Cite Y. Hinuma et al., *Computational Materials Science* **128**, 140 (2017), [doi:10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015), when using these paths. SeeK-path and spglib are installed dependencies, not copied source.
