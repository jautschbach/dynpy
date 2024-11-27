from numbers import Integral
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.components import connected_components
from distances import cartmag, modv, compute_atom_two, sym2mass

class Universe:
    def __init__(self, atom, **kwargs):
        self.atom = Atom(atom)
        #self.frame = Frame(frame)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def periodic(self, *args, **kwargs):
        return self.frame.is_periodic(*args, **kwargs)

    @property
    def orthorhombic(self):
        #print(self.frame.orthorhombic())
        return self.frame.orthorhombic()
    
    def compute_atom_two(self, *args, **kwargs):
        """
        Compute interatomic two body properties (e.g. bonds).

        Args:
            mapper (dict): Custom radii to use when determining bonds
            bond_extra (float): Extra additive factor to use when determining bonds
        """
        self.atom_two = compute_atom_two(self, *args, **kwargs)
    def compute_molecule(self):
        """Compute the :class:`~exatomic.molecule.Molecule` table."""
        self.molecule = compute_molecule(self)
        self.compute_molecule_count()
    def compute_atom_two(self, *args, **kwargs):
        """
        Compute interatomic two body properties (e.g. bonds).

        Args:
            mapper (dict): Custom radii to use when determining bonds
            bond_extra (float): Extra additive factor to use when determining bonds
        """
        self.atom_two = compute_atom_two(self, *args, **kwargs)
    def compute_unit_atom(self):
        """Compute minimal image for periodic systems."""
        self.unit_atom = UnitAtom.from_universe(self)
    
    def compute_molecule_com(self):
        cx, cy, cz = compute_molecule_com(self)
        self.molecule['cx'] = cx
        self.molecule['cy'] = cy
        self.molecule['cz'] = cz

    def compute_atom_count(self):
        """Compute number of atoms per frame."""
        self.frame['atom_count'] = self.atom.cardinal_groupby().size()

    def compute_molecule_count(self):
        """Compute number of molecules per frame."""
        self.frame['molecule_count'] = compute_molecule_count(self)

class Atom(pd.DataFrame):
    _index = 'atom'
    _cardinal = ('frame', np.int64)
    _categories = {'symbol': str, 'set': np.int64, 'molecule': np.int64,
                   'label': np.int64}
    _columns = ['x', 'y', 'z', 'symbol']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self[['x','y','z']] = self[['x','y','z']].astype(float)
        for key, value in kwargs.items():
            setattr(self, key, value)
    @property
    def nframes(self):
        """Return the total number of frames in the atom table."""
        return np.int64(self.frame.cat.as_ordered().max() + 1)
    
    @staticmethod
    def _determine_center(attr, coords):
        """Determine the center of the molecule with respect to
        the given attribute data. Used for the center of nuclear
        charge and center of mass."""
        center = 1/np.sum(attr)*np.sum(np.multiply(np.transpose(coords), attr), axis=1)
        center = pd.Series(center, index=['x', 'y', 'z'])
        return center
    
    def cardinal_groupby(self):
        """
        Group this object on it cardinal dimension (_cardinal).

        Returns:
            grpby: Pandas groupby object (grouped on _cardinal)
        """
        g, t = self._cardinal
        self[g] = self[g].astype(t)
        grpby = self.groupby(g)
        self[g] = self[g].astype('category')
        return grpby
    
    def get_element_masses(self):
        """Compute and return element masses from symbols."""
        return self['symbol'].astype('O').map(sym2mass)

    def get_atom_labels(self):
        """
        Compute and return enumerated atoms.

        Returns:
            labels (:class:`~exatomic.exa.core.numerical.Series`): Enumerated atom labels (of type int)
        """
        nats = self.cardinal_groupby().size().values
        labels = pd.Series([i for nat in nats for i in range(nat)], dtype='category')
        labels.index = self.index
        return labels
    
    def to_xyz(self, tag='symbol', header=False, comments='', columns=None,
               frame=None, units='Angstrom'):
        """
        Return atomic data in XYZ format, by default without the first 2 lines.
        If multiple frames are specified, return an XYZ trajectory format. If
        frame is not specified, by default returns the last frame in the table.

        Args:
            tag (str): column name to use in place of 'symbol'
            header (bool): if True, return the first 2 lines of XYZ format
            comment (str, list): comment(s) to put in the comment line
            frame (int, iter): frame or frames to return
            units (str): units (default angstroms)

        Returns:
            ret (str): XYZ formatted atomic data
        """
        # TODO :: this is conceptually a duplicate of XYZ.from_universe
        columns = (tag, 'x', 'y', 'z') if columns is None else columns
        frame = self.nframes - 1 if frame is None else frame
        if isinstance(frame, Integral): frame = [frame]
        if not isinstance(comments, list): comments = [comments]
        if len(comments) == 1: comments = comments * len(frame)
        df = self[self['frame'].isin(frame)].copy()
        if tag not in df.columns:
            if tag == 'Z':
                stoz = sym2z()
                df[tag] = df['symbol'].map(stoz)
        df['x'] *= 0.529177 #Length['au', units]
        df['y'] *= 0.529177 #Length['au', units]
        df['z'] *= 0.529177 #Length['au', units]
        grps = df.groupby('frame', observed=True)
        ret = ''
        formatter = {tag: '{:<5}'.format}
        stargs = {'columns': columns, 'header': False,
                  'index': False, 'formatters': formatter}
        t = 0
        for _, grp in grps:
            if not len(grp): continue
            tru = (header or comments[t] or len(frame) > 1)
            hdr = '\n'.join([str(len(grp)), comments[t], '']) if tru else ''
            ret = ''.join([ret, hdr, grp.to_string(**stargs), '\n'])
            t += 1
        return ret
    
class Frame(pd.DataFrame):
    """
    Information about the current frame; a frame is a concept that distinguishes
    atomic coordinates along a molecular dynamics simulation, geometry optimization,
    etc.

    +-------------------+----------+-------------------------------------------+
    | Column            | Type     | Description                               |
    +===================+==========+===========================================+
    | atom_count        | int      | non-unique integer (req.)                 |
    +-------------------+----------+-------------------------------------------+
    | molecule_count    | int      | non-unique integer                        |
    +-------------------+----------+-------------------------------------------+
    | ox                | float    | unit cell origin point in x               |
    +-------------------+----------+-------------------------------------------+
    | oy                | float    | unit cell origin point in y               |
    +-------------------+----------+-------------------------------------------+
    | oz                | float    | unit cell origin point in z               |
    +-------------------+----------+-------------------------------------------+
    | periodic          | bool     | true if periodic system                   |
    +-------------------+----------+-------------------------------------------+
    """
    _index = 'frame'
    _columns = ['atom_count']

#    @property
#    def _constructor(self):
#        return Frame

    def is_periodic(self, how='all'):
        """
        Check if any/all frames are periodic.

        Args:
            how (str): Require "any" frames to be periodic ("all" default)

        Returns:
            result (bool): True if any/all frame are periodic
        """
        if 'periodic' in self:
            if how == 'all' and np.all(self['periodic'] == True):
                return True
            elif how == 'any' and np.any(self['periodic'] == True):
                return True
        return False

    def is_variable_cell(self, how='all'):
        """
        Check if the simulation cell (applicable to periodic simulations) varies
        (e.g. variable cell molecular dynamics).
        """
        if self.is_periodic:
            if 'rx' not in self.columns:
                self.compute_cell_magnitudes()
            rx = self['rx'].min()
            ry = self['ry'].min()
            rz = self['rz'].min()
            if np.allclose(self['rx'], rx) and np.allclose(self['ry'], ry) and np.allclose(self['rz'], rz):
                return False
            else:
                return True
        raise PeriodicUniverseError()

    def add_cell_dm(self,celldm=None):
        """
        Add the unit cell dimensions to the frame table of the universe
        """
        for i, q in enumerate(("x", "y", "z")):
            for j, r in enumerate(("i", "j", "k")):
                if i == j:
                    self[q+r] = celldm
                else:
                    self[q+r] = 0.0
            self["o"+q] = 0.0
        self['periodic'] = True

    def compute_cell_magnitudes(self):
        """
        Compute the magnitudes of the unit cell vectors (rx, ry, rz).
        """
        self['rx'] = cartmag(self['xi'].values, self['yi'].values, self['zi'].values)
        self['ry'] = cartmag(self['xj'].values, self['yj'].values, self['zj'].values)
        self['rz'] = cartmag(self['xk'].values, self['yk'].values, self['zk'].values)

    def orthorhombic(self):
        if "xi" in self.columns and np.allclose(self["xj"], 0.0):
            return True
        return False


def compute_frame(universe):
    """
    Compute (minmal) :class:`~exatomic.frame.Frame` from
    :class:`~exatomic.container.Universe`.

    Args:
        uni (:class:`~exatomic.container.Universe`): Universe with atom table

    Returns:
        frame (:class:`~exatomic.frame.Frame`): Minimal frame table
    """
    return compute_frame_from_atom(universe.atom)


def compute_frame_from_atom(atom):
    """
    Compute :class:`~exatomic.frame.Frame` from :class:`~exatomic.atom.Atom`
    (or related).

    Args:
        atom (:class:`~exatomic.atom.Atom`): Atom table

    Returns:
        frame (:class:`~exatomic.frame.Frame`): Minimal frame table
    """
    atom = Atom(atom)
    frame = atom.cardinal_groupby().size().to_frame()
    frame.index = frame.index.astype(np.int64)
    frame.columns = ['atom_count']
    return Frame(frame)

class UnitAtom(pd.DataFrame):
    """
    In unit cell coordinates (sparse) for periodic systems. These coordinates
    are used to update the corresponding :class:`~exatomic.atom.Atom` object
    """
    _index = 'atom'
    _columns = ['x', 'y', 'z']

    #@property
    #def _constructor(self):
    #    return UnitAtom

    @classmethod
    def from_universe(cls, universe):
        if universe.periodic:
            if "rx" not in universe.frame.columns:
                universe.frame.compute_cell_magnitudes()
            a, b, c = universe.frame[["rx", "ry", "rz"]].max().values
            x = modv(universe.atom['x'].values, a)
            y = modv(universe.atom['y'].values, b)
            z = modv(universe.atom['z'].values, c)
            df = pd.DataFrame.from_dict({'x': x, 'y': y, 'z': z})
            df.index = universe.atom.index
            return cls(df[universe.atom[['x', 'y', 'z']] != df])
        raise PeriodicUniverseError()

class ProjectedAtom(pd.DataFrame):
    """
    Projected atom coordinates (e.g. on 3x3x3 supercell). These coordinates are
    typically associated with their corresponding indices in another dataframe.

    Note:
        This table is computed when periodic two body properties are computed;
        it doesn't have meaning outside of that context.

    See Also:
        :func:`~exatomic.two.compute_periodic_two`.
    """
    _index = 'two'
    _columns = ['x', 'y', 'z']

    #@property
    #def _constructor(self):
    #    return ProjectedAtom

class Molecule(pd.DataFrame):
    """
    Description of molecules in the atomic universe.
    """
    _index = 'molecule'
    _categories = {'frame': np.int64, 'formula': str, 'classification': object}

    #@property
    #def _constructor(self):
    #    return Molecule

    def classify(self, *classifiers):
        """
        Classify molecules into arbitrary categories.

        .. code-block:: Python

            u.molecule.classify(('solute', 'Na'), ('solvent', 'H(2)O(1)'))

        Args:
            classifiers: Any number of tuples of the form ('label', 'identifier', exact) (see below)

        Note:
            A classifier has 3 parts, "label", e.g. "solvent", "identifier", e.g.
            "H(2)O(1)", and exact (true or false). If exact is false (default),
            classification is greedy and (in this example) molecules with formulas
            "H(1)O(1)", "H(3)O(1)", etc. would get classified as "solvent". If,
            instead, exact were set to true, those molecules would remain
            unclassified.

        Warning:
            Classifiers are applied in the order passed; where identifiers overlap,
            the latter classification is used.

        See Also:
            :func:`~exatomic.algorithms.nearest.compute_nearest_molecules`
        """
        for c in classifiers:
            n = len(c)
            #if n != 3 and n != 2:
            #    raise ClassificationError()
        self['classification'] = None
        for classifier in classifiers:
            identifier = string_to_dict(classifier[0])
            classification = classifier[1]
            exact = classifier[2] if len(classifier) == 3 else False
            this = self
            for symbol, count in identifier.items():
                this = this[this[symbol] == count] if exact else this[this[symbol] >= 1]
            if len(this) > 0:
                self.loc[self.index.isin(this.index), 'classification'] = classification
            else:
                raise KeyError('No records found for {}, with identifier {}.'.format(classification, identifier))
        self['classification'] = self['classification'].astype('category')
        if len(self[self['classification'].isnull()]) > 0:
            print("Warning: Unclassified molecules remaining...")

    def get_atom_count(self):
        """
        Compute the number of atoms per molecule.
        """
        symbols = self._get_symbols()
        return self[symbols].sum(axis=1)

    def get_formula(self, as_map=False):
        """
        Compute the string representation of the molecule.
        """
        symbols = self._get_symbols()
        mcules = self[symbols].to_dict(orient='index')
        ret = map(dict_to_string, mcules.values())
        if as_map:
            return ret
        return list(ret)

    def _get_symbols(self):
        """
        Helper method to get atom symbols.
        """
        return [col for col in self if len(col) < 3 and col[0].istitle()]


def compute_molecule(universe):
    """
    Cluster atoms into molecules and create the :class:`~exatomic.molecule.Molecule`
    table.

    Args:
        universe: Atomic universe

    Returns:
        molecule: Molecule table

    Warning:
        This function modifies the universe's atom (:class:`~exatomic.atom.Atom`)
        table in place!
    """
    nodes = universe.atom.index.values
    bonded = universe.atom_two.loc[universe.atom_two['bond'] == True, ['atom0', 'atom1']]
    edges = zip(bonded['atom0'].astype(np.int64), bonded['atom1'].astype(np.int64))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # generate molecule indices for the atom table
    mapper = {}
    i = 0
    for k, v in g.degree():    # First handle single atom "molecules"
        if v == 0:
            mapper[k] = i
            i += 1
    for seht in connected_components(g):    # Second handle multi atom molecules
        for adx in seht:
            mapper[adx] = i
        i += 1
    universe.atom['molecule'] = universe.atom.index.map(lambda x: mapper[x])
    universe.atom['mass'] = universe.atom['symbol'].map(sym2mass).astype(float)
    grps = universe.atom.groupby('molecule')
    molecule = grps['symbol'].value_counts().unstack().fillna(0).astype(np.int64)
    molecule.columns.name = None
    molecule['mass'] = grps['mass'].sum()
    universe.atom['molecule'] = universe.atom['molecule'].astype('category')
    #del universe.atom['mass']
    return Molecule(molecule)


def compute_molecule_count(universe):
    """
    """
    if 'molecule' not in universe.atom.columns:
        universe.compute_molecule()
    #universe.atom._revert_categories()
    mapper = universe.atom.drop_duplicates('molecule').set_index('molecule')['frame']
    #universe.atom._set_categories()
    universe.molecule['frame'] = universe.molecule.index.map(lambda x: mapper[x])
    molecule_count = universe.molecule.groupby('frame').size()
    del universe.molecule['frame']
    return molecule_count


def compute_molecule_com(universe):
    """
    Compute molecules' centers of mass.
    """
    if 'molecule' not in universe.atom.columns:
        universe.compute_molecule()
    mass = universe.atom.get_element_masses()
    if universe.frame.is_periodic():
        xyz = universe.atom[['x', 'y', 'z']].copy()
        xyz.update(universe.visual_atom)
    else:
        xyz = universe.atom[['x', 'y', 'z']]
    xm = xyz['x'].mul(mass)
    ym = xyz['y'].mul(mass)
    zm = xyz['z'].mul(mass)
    #rm = xm.add(ym).add(zm)
    df = pd.DataFrame.from_dict({'xm': xm, 'ym': ym, 'zm': zm, 'mass': mass,
                                 'molecule': universe.atom['molecule']})
    groups = df.groupby('molecule')
    sums = groups.sum()
    cx = sums['xm'].div(sums['mass'])
    cy = sums['ym'].div(sums['mass'])
    cz = sums['zm'].div(sums['mass'])
    return cx, cy, cz

def string_to_dict(formula):
    """
    Convert string formula to a dictionary.

    Args:
        formula (str): String formula representation

    Returns:
        fdict (dict): Dictionary formula representation
    """
    obj = []
    if ')' not in formula and len(formula) <= 3 and all((not char.isdigit() for char in formula)):
        return {formula: 1}
    elif ')' not in formula:
        print('Incorrect formula syntax for {} (syntax example H(2)O(1)).'.format(formula))
    for s in formula.split(')'):
        if s != '':
            symbol, count = s.split('(')
            obj.append((symbol, np.int64(count)))
    return dict(obj)


def dict_to_string(formula):
    """
    Convert a dictionary formula to a string.

    Args:
        formula (dict): Dictionary formula representation

    Returns:
        fstr (str): String formula representation
    """
    return ''.join(('{0}({1})'.format(key.title(), formula[key]) for key in sorted(formula.keys()) if formula[key] > 0))