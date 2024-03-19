import numpy as np
import pandas as pd
import numba as nb
from platform import system
import isotopes

# For numba compiled functions
sysname= system().lower()
#nbpll = "linux" in sysname
nbpll = False
nbtgt = "parallel" if nbpll else "cpu"
nbche = not nbtgt

isotopedf = isotopes.as_df()
sym2z = isotopedf.drop_duplicates("symbol").set_index("symbol")["Z"].to_dict()
z2sym = {v: k for k, v in sym2z.items()}
sym2mass = {}
sym2radius = {}
sym2color = {}
for k, v in vars(isotopes).items():
    if isinstance(v, isotopes.Element):
        sym2mass[k] = v.mass
        sym2radius[k] = [v.cov_radius, v.van_radius]
        sym2color[k] = '#' + v.color[-2:] + v.color[3:5] + v.color[1:3]

@nb.vectorize(["float64(float64, float64, float64)"], nopython=True, target=nbtgt)
def cartmag(x, y, z):
    """
    Vectorized operation to compute the magnitude of a three component array.
    """
    return np.sqrt(x**2 + y**2 + z**2)

@nb.vectorize(["float64(float64, float64)"], nopython=True, target=nbtgt)
def modv(x, y):
    """
    Vectorized modulo operation.

    Args:
        x (array): 1D array
        y (array, float): Scalar or 1D array

    Returns:
        z (array): x modulo y
    """
    return np.mod(x, y)

class AtomTwo(pd.DataFrame):
    """Interatomic distances."""
    _index = "two"
    #_cardinal = ("frame", np.int64)
    _columns = ["atom0", "atom1", "dr"]
    _categories = {'symbols': str, 'atom0': np.int64, 'atom1': np.int64}

#    @property
#    def _constructor(self):
#        return AtomTwo

    @property
    def bonded(self):
        return self[self['bond'] == True]


class MoleculeTwo(pd.DataFrame):
    @property
    def _constructor(self):
        return MoleculeTwo


def compute_atom_two(universe, dmax=8.0, vector=False, bonds=True, **kwargs):
    """
    Compute interatomic distances and determine bonds.

    .. code-block:: python

        atom_two = compute_atom_two(uni, dmax=4.0)    # Max distance of interest as 4 bohr
        atom_two = compute_atom_two(uni, vector=True) # Return distance vector components as well as distance
        atom_two = compute_atom_two(uni, bonds=False) # Don't compute bonds
        # Compute bonds with custom covalent radii (atomic units)
        atom_two = compute_atom_two(unit, H=10.0, He=20.0, Li=30.0, bond_extra=100.0)

    Args:
        universe (:class:`~exatomic.core.universe.Universe`): Universe object with atom table
        dmax (float): Maximum distance of interest
        vector (bool): Compute distance vector (needed for angles)
        bonds (bool): Compute bonds (default True)
        kwargs: Additional keyword arguments for :func:`~exatomic.core.two._compute_bonds`
    """
    if universe.periodic:
        if universe.orthorhombic and vector:
            atom_two = compute_pdist_ortho(universe, dmax=dmax)
        #elif universe.orthorhombic:
            #atom_two = compute_pdist_ortho_nv(universe, dmax=dmax)
        else:
            raise NotImplementedError("Only supports orthorhombic cells")
    #elif vector:
    #    atom_two = compute_pdist(universe, dmax=dmax)
    else:
        #atom_two = compute_pdist_nv(universe, dmax=dmax)
        raise NotImplementedError("Cell must be periodic")
    if bonds:
        _compute_bonds(universe.atom, atom_two, **kwargs)
    return atom_two


def compute_pdist_ortho(universe, dmax=8.0):
    """
    Compute interatomic distances between atoms in an orthorhombic
    periodic cell.

    Args:
        universe (:class:`~exatomic.core.universe.Universe`): A universe
        bonds (bool): Compute bonds as well as distances
        bond_extra (float): Extra factor to use when determining bonds
        dmax (float): Maximum distance of interest
        rtol (float): Relative tolerance (float equivalence)
        atol (float): Absolute tolerance (float equivalence)
        radii (kwargs): Custom (covalent) radii to use when determining bonds
    """
    if "rx" not in universe.frame.columns:
        universe.frame.compute_cell_magnitudes()
    dxs = []
    dys = []
    dzs = []
    drs = []
    atom0s = []
    atom1s = []
    prjs = []
    atom = universe.atom[["x", "y", "z", "frame"]].copy()
    atom.update(universe.unit_atom)

    for fdx, group in atom.groupby("frame"):
        if len(group) > 0:
            a, b, c = universe.frame.loc[fdx, ["rx", "ry", "rz"]]
            values = pdist_ortho(group['x'].values.astype(float),
                                 group['y'].values.astype(float),
                                 group['z'].values.astype(float),
                                 a, b, c,
                                 group.index.values.astype(int), dmax)
            dxs.append(values[0])
            dys.append(values[1])
            dzs.append(values[2])
            drs.append(values[3])
            atom0s.append(values[4])
            atom1s.append(values[5])
            prjs.append(values[6])
    dxs = np.concatenate(dxs)
    dys = np.concatenate(dys)
    dzs = np.concatenate(dzs)
    drs = np.concatenate(drs)
    atom0s = np.concatenate(atom0s)
    atom1s = np.concatenate(atom1s)
    prjs = np.concatenate(prjs)
    return AtomTwo.from_dict({'dx': dxs, 'dy': dys, 'dz': dzs, 'dr': drs,
                              'atom0': atom0s, 'atom1': atom1s, 'projection': prjs})
@nb.jit(nopython=True, nogil=True, parallel=nbpll)
def pdist_ortho(ux, uy, uz, a, b, c, index, dmax=8.0):
    """
    Pairwise two body calculation for bodies in an orthorhombic periodic cell.

    Does return distance vectors.

    An orthorhombic cell is defined by orthogonal vectors of length a and b
    (which define the base) and height vector of length c. All three vectors
    intersect at 90° angles. If a = b = c the cell is a simple cubic cell.
    This function assumes the unit cell is constant with respect to an external
    frame of reference and that the origin of the cell is at (0, 0, 0).

    Args:
        ux (array): In unit cell x array
        uy (array): In unit cell y array
        uz (array): In unit cell z array
        a (float): Unit cell dimension a
        b (float): Unit cell dimension b
        c (float): Unit cell dimension c index (array): Atom indexes
        dmax (float): Maximum distance of interest
    """
    m = [-1, 0, 1]
    dmax2 = dmax**2
    n = len(ux)
    nn = n*(n - 1)//2
    dx = np.empty((nn, ), dtype=np.float64)
    dy = dx.copy()
    dz = dx.copy()
    dr = dx.copy()
    ii = np.empty((nn, ), dtype=np.int64)
    jj = ii.copy()
    projection = ii.copy()
    k = 0
    # For each atom i
    for i in range(n):
        xi = ux[i]
        yi = uy[i]
        zi = uz[i]
        # For each atom j
        for j in range(i+1, n):
            xj = ux[j]
            yj = uy[j]
            zj = uz[j]
            dpr = dmax2
            inck = False
            prj = 0
            # Check all projections of atom i
            # Note that i, j are in the unit cell so we make a 3x3x3 'supercell'
            # of i around j
            # The index of the projections of i go from 0 to 26 (27 projections)
            # The 13th projection is the unit cell itself.
            for aa in m:
                for bb in m:
                    for cc in m:
                        pxi = xi + aa*a
                        pyi = yi + bb*b
                        pzi = zi + cc*c
                        dpx_ = pxi - xj
                        dpy_ = pyi - yj
                        dpz_ = pzi - zj
                        dpr_ = dpx_**2 + dpy_**2 + dpz_**2
                        # The second criteria here enforces that prefer the projection
                        # with the largest value (i.e. 0 = [-1, -1, -1] < 13 = [0, 0, 0]
                        # < 26 = [1, 1, 1])
                        # The system sets a fixed preference for the projected positions rather
                        # than having a random choice.
                        if dpr_ < dpr:
                            dx[k] = dpx_
                            dy[k] = dpy_
                            dz[k] = dpz_
                            dr[k] = np.sqrt(dpr_)
                            ii[k] = index[i]
                            jj[k] = index[j]
                            projection[k] = prj
                            dpr = dpr_
                            inck = True
                        prj += 1
            if inck:
                k += 1
    dx = dx[:k]
    dy = dy[:k]
    dz = dz[:k]
    dr = dr[:k]
    ii = ii[:k]
    jj = jj[:k]
    projection = projection[:k]
    return dx, dy, dz, dr, ii, jj, projection

def _compute_bonds(atom, atom_two, bond_extra=0.45, **radii):
    """
    Compute bonds inplce.

    Args:
        bond_extra (float): Additional amount for determining bonds
        radii: Custom radii to use for computing bonds
    """
    atom['symbol'] = atom['symbol'].astype('category')
    radmap = {sym: sym2radius[sym][0] for sym in atom['symbol'].cat.categories}
    radmap.update(radii)
    maxdr = (atom_two['atom0'].map(atom['symbol']).map(radmap).astype(float) +
             atom_two['atom1'].map(atom['symbol']).map(radmap).astype(float) + bond_extra)
    atom_two['bond'] = np.where(atom_two['dr'] <= maxdr, True, False)