import pandas as pd
import numpy as np
import scipy as sp
import os
import exa
import exatomic
from exatomic import qe
from neighbors_input import *
from exatomic.algorithms import neighbors
from exatomic.algorithms.neighbors import periodic_nearest_neighbors_by_atom    # Only valid for simple cubic periodic cells
from nuc import *

def My_fn(x,analyte,symbols):
    solv = [s for s in list(set(symbols)) if s is not analyte]
    sym = x.split()[0]
    if sym == analyte:
        return 0
    else:
        for i,s in enumerate(solv):
            if sym == s:
                return i+1


def gen_inputs(traj_dir=traj_dir):
    nuc_df=pd.DataFrame.from_dict(nuc)
    trajs = [t.name for t in os.scandir(traj_dir) if t.name.isnumeric()]
    print("Parsing "+ str(len(trajs)) + " trajectories...")
    for traj in trajs:
        print("Trajectory "+traj+'...')
        pos_dir = traj_dir + traj + '/'
        xyzs = pos_dir + "xyzs/"
        pos = list(filter(lambda x: ".pos" in x, os.listdir(pos_dir)))[0]
        atom = qe.parse_xyz(pos_dir+'/'+pos,symbols=symbols)
        atom['frame'] = atom['frame'].astype(int)
        atom = atom[(atom['frame']>=atom.frame[0] + start_prod) & (atom['frame']%sample_freq==0)]

        u = exatomic.Universe()
        u.atom = atom

        # Add the unit cell dimensions to the frame table of the universe
        for i, q in enumerate(("x", "y", "z")):
            for j, r in enumerate(("i", "j", "k")):
                if i == j:
                    u.frame[q+r] = celldm
                else:
                    u.frame[q+r] = 0.0
            u.frame["o"+q] = 0.0
        u.frame['periodic'] = True

        grouped = u.atom.groupby('frame')

        if write_ADF:
            ADF_dir = pos_dir+"ADF/"
            if not os.path.isdir(ADF_dir):
                os.mkdir(ADF_dir)
            solute = "".join([n for n in nuc_symbol if n.isalpha()])
            if not skip_compute_neighbors:
                print("Computing nearest neighbors...")
                dct = neighbors.periodic_nearest_neighbors_by_atom(u,    # Universe with all frames from which we want to extract clusters
                                        solute,       # Source atom from which we will search for neighbors
                                        celldm,       # Cubic cell dimension
                                        [nn],         # Cluster sizes we want
                                        #take_prj=14,
                                        dmax=celldm/2)

                grouped_nn = dct[nn].atom.groupby('frame')
                if write_xyzs:
                    print("Writing .xyz files and ADF inputs...")
                    if not os.path.isdir(xyzs):
                        os.mkdir(xyzs)
                else:
                    print("Writing ADF inputs...")
            else:
                print("Writing ADF inputs with coordinates from previously computed clusters...")

        if write_GIPAW:
            print("Writing QE-GIPAW inputs...")
            paw_dir = pos_dir + "GIPAW/"
            if not os.path.isdir(paw_dir):
                os.mkdir(paw_dir)

        i = 0
        dt = timestep*2.418884e-5
        for frame, group in grouped:
            i+=1
            fname = str(i).zfill(4)
            time = frame*dt
            comment = "frame: {}, time: {}".format(frame, time)

            if write_ADF:
                if not skip_compute_neighbors:
                    cluster = grouped_nn.get_group(frame).copy()
                    cluster_angs = cluster.copy()
                    cluster_angs.loc[:,'x'] = cluster.loc[:,'x']*0.529177
                    cluster_angs.loc[:,'y'] = cluster.loc[:,'y']*0.529177
                    cluster_angs.loc[:,'z'] = cluster.loc[:,'z']*0.529177

                    shit = cluster_angs.loc[:,['symbol','x','y','z']].values
                    a = str(shit).replace('[','')
                    b = a.replace(']','')
                    c = b.replace('\n ', '\n')
                    xyz = c.replace("\'",'')
                    xyz_lines = xyz.split('\n')
                    sorted_xyz = sorted(xyz_lines,key=lambda x: My_fn(x,solute,symbols))
                    coord = "\n".join(sorted_xyz)
                    nat = len(cluster)

                    if write_xyzs:
                        with open(xyzs + str(i).zfill(4)+'.xyz', 'w') as f:
                            f.write(str(nat)+"\n{0}\n".format(comment)+coord)
                else:
                    with open(xyzs + str(i).zfill(4)+'.xyz', 'r') as f:
                        lines = f.readlines()
                    coord = "".join(lines[2:])
                    nat = len(lines[2:])


                if not os.path.isdir(ADF_dir + fname):
                    os.mkdir(ADF_dir +"/"+ fname)

                ch = 0
                clines = coord.split('\n')
                #print(clines)
                for line in clines:
                    ll=line.split()
                    if ll[0] == solute:
                        ch += solute_charge
                    if ll[0] == 'O':
                        ch += -2
                    if ll[0] == 'H':
                        ch += 1

                with open(ADF_dir + fname + "/" + fname + "-scf.inp", 'w') as g:
                    g.write(ADF_in.format(coord,comment,ch,solute))


                with open(ADF_dir + fname + "/" + fname + ".slm", 'w') as g:
                    g.write(ADF_slm.format(traj, fname, CCR_scratch))

            if write_GIPAW:
                #print(atom.head())
                #print(nuc_df)
                atom_angs = atom.copy()
                atom_angs.loc[:,'x'] = atom.loc[:,'x']*0.529177
                atom_angs.loc[:,'y'] = atom.loc[:,'y']*0.529177
                atom_angs.loc[:,'z'] = atom.loc[:,'z']*0.529177
                shit = atom_angs[atom_angs['frame']==frame].loc[:,['symbol','x','y','z']].values
                a = str(shit).replace('[','')
                b = a.replace(']','')
                c = b.replace('\n ', '\n')
                full_coord = c.replace("\'",'')
                nat = len(shit)

                if not os.path.isdir(paw_dir + fname):
                    os.mkdir(paw_dir +"/"+ fname)
                with open(paw_dir + fname + "/" + fname + "-scf.inp", 'w') as g:
                    g.write(PAW_scf_in.format(comment,fname,celldm,nat,full_coord))

                with open(paw_dir + fname + "/" + fname + "-efg.inp", 'w') as g:
                    g.write(PAW_efg_in.format(comment,fname,str(nuc_df.loc['Q',nuc_symbol]*1e30)))
        #Your slurm template here!
                with open(paw_dir + fname + "/" + fname + ".slm", 'w') as g:
                    g.write(PAW_slm.format(fname,traj,CCR_scratch))
    print("Done")
    return(dct[nn])
