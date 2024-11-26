import pandas as pd
import numpy as np
import scipy as sp
import os
import sys
#import exa
#import exatomic
#exa.logging.disable(level=10)
#exa.logging.disable(level=20)
#from exatomic import qe
#from exatomic.algorithms import neighbors
#from exatomic.algorithms.neighbors import periodic_nearest_neighbors_by_atom    # Only valid for simple cubic periodic cells
from nuc import *
from parseMD import *
from xyz import *
from universe import *
import signal
from collections import defaultdict

#signal.signal(signal.SIGINT, signal_handler)

def periodic_nearest_neighbors_by_atom(uni, source, a, sizes, **kwargs):
    """
    Determine nearest neighbor molecules to a given source (or sources) and
    return the data as a dataframe.

    Warning:
        For universes with more than about 250 atoms, consider using the
        slower but more memory efficient
        :func:`~exatomic.algorithms.neighbors.periodic_nearest_neighbors_by_atom_large`.

    For a simple cubic periodic system with unit cell dimension ``a``,
    clusters can be generated as follows. In the example below, additional
    keyword arguments have been included as they are almost always required
    in order to correctly identify molecular units semi-empirically.

    .. code-block:: python

        periodic_nearest_neighbors_by_atom(u, [0], 40.0, [0, 5, 10, 50],
                                           dmax=40.0, C=1.6, O=1.6)

    Argument descriptions can be found below. The additional keyword arguments,
    ``dmax``, ``C``, ``O``, are passed directly to the two body computation used
    to determine (semi-empirically) molecular units. Note that although molecules
    are computed, neighboring molecular units are determine by an atom to atom
    criteria.

    Args:
        uni (:class:`~exatomic.core.universe.Universe`): Universe
        source (int, str, list): Integer label or string symbol of source atom
        a (float): Cubic unit cell dimension
        sizes (list): List of slices to create
        kwargs: Additional keyword arguments to be passed to atom two body calculation

    Returns:
        dct (dict): Dictionary of sliced universes and nearest neighbor table

    See Also:
        Sliced universe construction can be facilitated by
        :func:`~exatomic.algorithms.neighbors.construct`.
    """
    def sorter(group, source_atom_idxs):
        s = group[['atom0', 'atom1']].stack()
        return s[~s.isin(source_atom_idxs)].reset_index()

    if "label" not in uni.atom.columns:
        uni.atom['label'] = uni.atom.get_atom_labels()
    dct = defaultdict(list)
    grps = uni.atom.groupby("frame")
    ntot = len(grps)
    #fp = FloatProgress(description="Slicing:")
    #display(fp)
    for i, (fdx, atom) in enumerate(grps):
        if len(atom) > 0:
            #print(fdx,atom)
            uu = _create_super_universe(Universe(atom=atom.copy()), a)
            uu.frame = compute_frame_from_atom(uu.atom) #AP
            uu.frame.add_cell_dm(celldm = a*3) #AP
            uu.compute_unit_atom() #AP
            uu.compute_atom_two(vector=True,**kwargs)
            uu.compute_molecule()
            if isinstance(source, (int, np.int32, np.int64)):
                source_atom_idxs = uu.atom[(uu.atom.index.isin([source])) &
                                           (uu.atom['prj'] == 13)].index.values
            elif isinstance(source, (list, tuple)):
                source_atom_idxs = uu.atom[uu.atom['label'].isin(source) &
                                           (uu.atom['prj'] == 13)].index.values
            else:
                source_atom_idxs = uu.atom[(uu.atom['symbol'] == source) &
                                           (uu.atom['prj'] == 13)].index.values
            source_molecule_idxs = uu.atom.loc[source_atom_idxs, 'molecule'].unique().astype(int)
            uu.atom_two['frame'] = uu.atom_two['atom0'].map(uu.atom['frame'])
            nearest_atoms = uu.atom_two[(uu.atom_two['atom0'].isin(source_atom_idxs)) |
                                        (uu.atom_two['atom1'].isin(source_atom_idxs))].sort_values("dr")[['frame', 'atom0', 'atom1']]
            nearest = nearest_atoms.groupby("frame").apply(sorter, source_atom_idxs=source_atom_idxs)
            del nearest['level_1']
            nearest.index.names = ['frame', 'idx']
            nearest.columns = ['two', 'atom']
            nearest['molecule'] = nearest['atom'].map(uu.atom['molecule'])
            nearest = nearest[~nearest['molecule'].isin(source_molecule_idxs)]
            nearest = nearest.drop_duplicates('molecule', keep='first')
            nearest.reset_index(inplace=True)
            nearest['frame'] = nearest['frame'].astype(int)
            nearest['molecule'] = nearest['molecule'].astype(int)
            dct['nearest'].append(nearest)
            for nn in sizes:
                atm = []
                for j, fdx in enumerate(nearest['frame'].unique()):
                    mdxs = nearest.loc[nearest['frame'] == fdx, 'molecule'].tolist()[:nn]
                    mdxs.append(source_molecule_idxs[j])
                    atm.append(uu.atom[uu.atom['molecule'].isin(mdxs)][['symbol', 'x', 'y', 'z', 'frame']].copy())
                dct[nn].append(pd.concat(atm, ignore_index=True))
        #fp.value = i/ntot*100
    dct['nearest'] = pd.concat(dct['nearest'], ignore_index=True)
    for nn in sizes:
        dct[nn] = Universe(atom=pd.concat(dct[nn], ignore_index=True))
    #fp.close()
    return dct

def _create_super_universe(u, a):
    """
    Generate a 3x3x3 super cell from a cubic periodic universe

    Args:
        u (:class:`~exatomic.core.universe.Universe`): Universe
        a (float): Cubic unit cell dimension

    Returns:
        uni (:class:`~exatomic.core.universe.Universe`): Universe of 3x3x3x super cell
    """
    adxs = []
    xs = []
    ys = []
    zs = []
    prjs = []
    fdxs = []
    #print(u.atom)
    grps = u.atom.groupby("frame")
    for fdx, atom in grps:
        adx, x, y, z, prj = _worker(atom.index.values.astype(np.int64),
                                    atom['x'].values.astype(np.float64),
                                    atom['y'].values.astype(np.float64),
                                    atom['z'].values.astype(np.float64), a)
        adxs.append(adx)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        prjs.append(prj)
        fdxs += [fdx]*len(adx)
    adxs = np.concatenate(adxs)
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    zs = np.concatenate(zs)
    prjs = np.concatenate(prjs)
    # Overwrite the last 'atom' group because that value doesn't need to exist anymore
    atom = pd.DataFrame.from_dict({'atom': adxs, 'x': xs, 'y': ys, 'z': zs, 'prj': prjs})
    atom['frame'] = fdxs
    atom['symbol'] = atom['atom'].map(u.atom['symbol'])
    atom['label'] = atom['atom'].map(u.atom['label'])
    atom = Atom(atom)
    #print(atom)
    #nat = len(atom[(atom.frame == 0) & (atom.prj == 0)])
    #print(atom.frame.unique())
    #nat_frames = [nat]*len(atom.frame.unique())
    #frame = pd.DataFrame(nat_frames,index = atom.frame.unique(),columns=['atom_count'])
    #frame['periodic'] = True
    #print(frame)
    return Universe(atom=atom)

def _worker(idx, x, y, z, a):
    """
    Generate a 3x3x3 'super' cell from a cubic unit cell.

    Args:
        idx (array): Array of index values
        x (array): Array of x coordinates
        y (array): Array of y coordinates
        z (array): Array of z coordinates
        a (float): Cubic unit cell dimension
    """
    n = len(x)
    idxs = np.empty((27*n, ), dtype=np.int64)
    prj = idxs.copy()
    px = np.empty((27*n, ), dtype=np.float64)
    py = px.copy()
    pz = px.copy()
    p = 0
    m = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                for l in range(n):
                    idxs[m] = idx[l]
                    px[m] = x[l] + i*a
                    py[m] = y[l] + j*a
                    pz[m] = z[l] + k*a
                    prj[m] = p
                    m += 1
                p += 1
    return idxs, px, py, pz, prj

def My_fn(x,analyte,symbols):
    solv = [s for s in list(set(symbols)) if s is not analyte]
    sym = x.split()[0]
    if sym == analyte:
        #print(sym)
        return 0
    else:
        for i,s in enumerate(solv):
            if sym == s:
                return i+1


def gen_inputs(dynpy_params):
    nuc_df=pd.DataFrame.from_dict(nuc)
    PD = dynpy_params.ParseDynamics
    ss = dynpy_params.Snapshots
    it = dynpy_params.InputTemplates

    if 'write_xyzs' not in ss.__dict__.keys():
        ss.write_xyzs = False
    if 'scratch' not in ss.__dict__.keys():
        ss.scratch = "/scratch/"

    print("Parsing "+ str(len(PD.trajs)) + " trajectories...")
    for traj in PD.trajs:
        print("Trajectory "+traj+'...')
        pos_dir = PD.traj_dir + traj + '/'
        if (ss.write_ADF and ss.skip_compute_neighbors) != True:
            u, vel = PARSE_MD(PD,pos_dir)

        if ss.write_GIPAW:
            print("Writing QE-GIPAW inputs...")
            paw_dir = pos_dir + "GIPAW/"
            if not os.path.isdir(paw_dir):
                os.mkdir(paw_dir)
            if ss.write_xyzs:
                paw_xyzs = paw_dir + "xyzs/"
                if not os.path.isdir(paw_xyzs):
                    os.mkdir(paw_xyzs)
            grouped = u.atom.groupby('frame')
            ntyp = len(u.atom.symbol.unique())

            for i, (frame,group) in enumerate(grouped):
                fname = str(i).zfill(4)
                time = frame*PD.timestep
                coord = Atom(group).to_xyz(frame=frame)          
                nat = len(Atom(group))
                comment = "frame: {}, time: {}".format(frame, time)
                if ss.write_xyzs:
                    with open(paw_xyzs + fname.zfill(4)+'.xyz', 'w') as f:
                        f.write(str(nat)+"\n{0}\n".format(comment)+coord)
    
                if not os.path.isdir(paw_dir + fname):
                    os.mkdir(paw_dir +"/"+ fname)
                with open(paw_dir + fname + "/" + fname + "-scf.inp", 'w') as g:
                    g.write(it.PAW_scf_in.format(comment,fname,PD.celldm,nat,ntyp,coord))

                with open(paw_dir + fname + "/" + fname + "-efg.inp", 'w') as g:
                    g.write(it.PAW_efg_in.format(comment,fname,ss.scratch,traj))
                with open(paw_dir + fname + "/" + fname + ".slm", 'w') as g:
                    g.write(it.PAW_slm.format(fname,traj,ss.scratch))


        if ss.write_ADF:
            print("Generating clusters and writing ADF inputs...")
            ADF_dir = pos_dir+"ADF/"
            if not os.path.isdir(ADF_dir):
                os.mkdir(ADF_dir)
            adf_xyzs = ADF_dir + "xyzs/"

            if 'analyte_label' not in ss.__dict__.keys():
                #print("analyte_label not provided. Inferring from nuc_symbol...")
                #try:
                #    solute = "".join([n for n in ss.nuc_symbol if n.isalpha()])
                #except:
                #    print("Unable to infer analyte_label")
                #    sys.exit(2)
                print("Error: Missing required input variable analyte_label in class Snapshots")
                sys.exit(2)
            if 'skip_compute_neighbors' not in ss.__dict__.keys():
                ss.skip_compute_neighbors = False
            if 'analyte_symbol' not in ss.__dict__.keys():
                print("Error: Missing required input variable analyte_symbol in class Snapshots")
                sys.exit(2)
            if 'solute_charge' not in ss.__dict__.keys():
                print("Error: Missing required input variable solute_charge in class Snapshots")
                sys.exit(2)
            if 'formal_charges' not in ss.__dict__.keys():
                print("Error: Missing required input variable formal_charges in class Snapshots")
                sys.exit(2)
            if 'nn' not in ss.__dict__.keys():
                ss.nn = 30
                print("Warning: Number of nearest neighbors (nn) not provided. Defaulting to nn=30")          

            if not ss.skip_compute_neighbors:
                print("Computing nearest neighbors...")
                dct = periodic_nearest_neighbors_by_atom(u,    # Universe with all frames from which we want to extract clusters
                                        ss.analyte_label,       # Source atom from which we will search for neighbors
                                        PD.celldm,       # Cubic cell dimension
                                        [ss.nn],         # Cluster sizes we want
                                        #take_prj=14,
                                        dmax=PD.celldm/2)

                nn_uni = dct[ss.nn]
                nn_atom_grouped = nn_uni.atom.groupby('frame')
            else:
                print("Writing ADF inputs with coordinates from previously computed clusters...")
                try:
                    atoms = []
                    xyz_files = [x.name for x in os.scandir(adf_xyzs) if x.name.split('.')[0].isnumeric()]
                    for i,xyz_file in enumerate(xyz_files):
                        with open(adf_xyzs + xyz_file, 'r') as f:
                            lines = f.readlines()
                        frame = int(lines[1].split()[1].strip(','))
                        atom = XYZ(adf_xyzs + xyz_file).atom
                        atom['frame'] = frame
                        atoms.append(atom)
                    nn_uni = Universe(atom = pd.concat(atoms))
                    nn_atom_grouped = nn_uni.atom.groupby('frame')
                except:
                    print("No precomputed xyz clusters were found. Set ss.skip_compute_neighbors to False to generate clusters.")
                    sys.exit(2)
            if ss.write_xyzs:
                if not os.path.isdir(adf_xyzs):
                    os.mkdir(adf_xyzs)

            for i, (frame,group) in enumerate(nn_atom_grouped):
                fname = str(i).zfill(4)
                time = frame*PD.timestep
                coord = Atom(group).to_xyz(frame=frame)
                #sorted_xyz = sorted(xyz_lines,key=lambda x: My_fn(x,solute_sym,symbols))
                nat = len(Atom(group))
                comment = "frame: {}, time: {}".format(frame, time)
                if ss.write_xyzs:
                    with open(adf_xyzs + fname + '.xyz', 'w') as f:
                        f.write(str(nat)+"\n{0}\n".format(comment)+coord)
    
                ch = ss.solute_charge
                cluster_labels = group.symbol.values #will this work if coord read from xyzs?
                for lab in cluster_labels:
                    if lab != ss.analyte_symbol:
                        ch+=ss.formal_charges[lab]
                if not os.path.isdir(ADF_dir + fname):
                    os.mkdir(ADF_dir +"/"+ fname)

                with open(ADF_dir + fname + "/" + fname + "-scf.inp", 'w') as g:
                    g.write(it.ADF_in.format(coord,comment,ch,ss.analyte_symbol))

                with open(ADF_dir + fname + "/" + fname + ".slm", 'w') as g:
                    g.write(it.ADF_slm.format(traj, fname, ss.analyte_symbol))
        
    print("Done")
