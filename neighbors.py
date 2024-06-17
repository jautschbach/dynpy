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
import signal
from dynpy import signal_handler

#signal.signal(signal.SIGINT, signal_handler)

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
        write_xyzs = False
    if 'nn' not in ss.__dict__.keys():
        nn = 30
    if 'skip_compute_neighbors' not in ss.__dict__.keys():
        skip_compute_neighbors = False
    trajs = parse_many(PD)    #Get list of trajectory directories to parse
    
    print("Parsing "+ str(len(trajs)) + " trajectories...")
    for traj in trajs:
        print("Trajectory "+traj+'...')
        pos_dir = PD.traj_dir + traj + '/'
        if (ss.write_ADF and ss.skip_compute_neighbors) != True:
            u, vel = PARSE_MD(pos_dir, PD)

        if ss.write_GIPAW:
            print("Writing QE-GIPAW inputs...")
            paw_dir = pos_dir + "GIPAW/"
            if not os.path.isdir(paw_dir):
                os.mkdir(paw_dir)
            if ss.write_xyzs:
                xyzs = pos_dir + "xyzs/"
                if not os.path.isdir(xyzs):
                    os.mkdir(xyzs)
            grouped = u.atom.groupby('frame')
            ntyp = len(u.atom.symbol.unique())

            for i, (frame,group) in enumerate(grouped):
                fname = str(i).zfill(4)
                time = frame*PD.timestep
                coord = exatomic.Atom(group).to_xyz(frame=frame)          
                nat = len(exatomic.Atom(group))
                comment = "frame: {}, time: {}".format(frame, time)
                if ss.write_xyzs:
                    with open(xyzs + fname.zfill(4)+'.xyz', 'w') as f:
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
            xyzs = ADF_dir + "xyzs/"

            if 'analyte_label' in ss.__dict__.keys():
                solute = ss.analyte_label
            else:
                print("analyte_label not provided. Inferring from nuc_symbol...")
                try:
                    solute = "".join([n for n in ss.nuc_symbol if n.isalpha()])
                except:
                    print("Unable to infer analyte_label")
                    sys.exit(2)
            
            if not ss.skip_compute_neighbors:
                print("Computing nearest neighbors...")
                dct = neighbors.periodic_nearest_neighbors_by_atom(u,    # Universe with all frames from which we want to extract clusters
                                        solute,       # Source atom from which we will search for neighbors
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
                    xyz_files = [x.name for x in os.scandir(xyzs) if x.name.split('.')[0].isnumeric()]
                    for i,xyz_file in enumerate(xyz_files):
                        with open(xyzs + xyz_file, 'r') as f:
                            lines = f.readlines()
                        frame = int(lines[1].split()[1].strip(','))
                        atom = exatomic.XYZ(xyzs + xyz_file).atom
                        atom['frame'] = frame
                        atoms.append(atom)
                    nn_uni = exatomic.Universe(atom = pd.concat(atoms))
                    nn_atom_grouped = nn_uni.atom.groupby('frame')
                except:
                    print("No precomputed xyz clusters were found. Set ss.skip_compute_neighbors to False to generate clusters.")
                    sys.exit(2)
            if ss.write_xyzs:
                if not os.path.isdir(xyzs):
                    os.mkdir(xyzs)

            for i, (frame,group) in enumerate(nn_atom_grouped):
                fname = str(i).zfill(4)
                time = frame*PD.timestep
                coord = exatomic.Atom(group).to_xyz(frame=frame)
                #sorted_xyz = sorted(xyz_lines,key=lambda x: My_fn(x,solute_sym,symbols))
                nat = len(exatomic.Atom(group))
                comment = "frame: {}, time: {}".format(frame, time)
                if ss.write_xyzs:
                    with open(xyzs + fname + '.xyz', 'w') as f:
                        f.write(str(nat)+"\n{0}\n".format(comment)+coord)
    
                ch = ss.solute_charge
                cluster_labels = group.symbol.values #will this work if coord read from xyzs?
                for lab in cluster_labels:
                    if lab != solute:
                        ch+=ss.formal_charges[lab]
                if not os.path.isdir(ADF_dir + fname):
                    os.mkdir(ADF_dir +"/"+ fname)

                with open(ADF_dir + fname + "/" + fname + "-scf.inp", 'w') as g:
                    g.write(it.ADF_in.format(coord,comment,ch,solute))

                with open(ADF_dir + fname + "/" + fname + ".slm", 'w') as g:
                    g.write(it.ADF_slm.format(traj, fname, ss.scratch))
        
    print("Done")
