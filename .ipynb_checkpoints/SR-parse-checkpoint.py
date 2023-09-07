import pandas as pd
import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy import signal
from scipy import constants
import os
import sys
import exa
import exatomic
#exa.logging.disable(level=10)
#exa.logging.disable(level=20)
from exatomic import qe
import math
from parse import *
from SRrax import *
import gc
from multiprocessing import Pool, cpu_count, set_start_method
#import ray
import time
start_time = time.time()

traj = 'ss'
pres = 'pp'
temp = '353K'
prefix = 'water'
celldm = {'9kpa': 476.68,
          '15kpa': 402.04,
          '34kpa': 306.06,
          '47kpa': 274.75}
#celldm = {'2kpa':  794.34,
#          '10kpa': 464.53,
#          '18kpa': 381.87,
#          '42kpa': 287.91,
#          '69kpa': 244.00}
#celldm = {'11kpa': 454.09,
#          '21kpa': 366.05,
#          '60kpa': 257.96,
#          '101kpa': 216.86}
path = "/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/"+temp+"/"+pres+"/"+traj+"/"
#path = "/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/"+temp+"/"+pres+"/NVT/"+traj+"/"
#atom = qe.parse_xyz(scratch + traj+'.pos', symbols=(['O']*64+['H']*128))
dt = 0.2
atom_start = 301
atom_end = 5300
vel_start = 301
vel_end = 5300
mod = 1
#atom_nframes = (atom_end-atom_start)//mod
#vel_nframes = (vel_end-vel_start)//mod
nat = 600
nmol = 200
mol_type='water'
atom = parse_tinker_atom(path,prefix,nat,atom_start,atom_end,mod)
vel = parse_tinker_vel(path,prefix,nat,vel_start,vel_end,mod)
#atom = atom[atom['frame']<=vel.frame.max()]
#atom = pd.read_csv(path+"water-"+temp+"-"+pres+"-0.2ps-dt-"+traj+"-atom-table.csv",dtype={'symbol':'category','x':'f8','y':'f8','z':'f8','frame':'i8'},skiprows=range(1,atom_start*nat+1),nrows=(atom_end-atom_start+1)*nat,header=0)
#atom['frame'] = atom['frame']-atom['frame'].min()
#vel = pd.read_csv(path+"water-"+temp+"-"+pres+"-0.2ps-dt-"+traj+"-vel.csv",dtype={'symbol':'category','x':'f8','y':'f8','z':'f8','frame':'i8'},skiprows=range(1,vel_start*nat+1),nrows=(vel_end-vel_start+1)*nat,header=0)
#vel['frame'] = vel['frame']-vel['frame'].min()
print("parse tinker--- %s seconds ---" % (time.time() - start_time))

#print(atom.tail().values)
#atom.to_csv(scratch+"/test-atom.csv")
#atom.loc[:,'frame'] = atom.frame.values.astype(int)
#atom = atom[atom['frame']%mod==0]
#vel = vel[vel['frame']%mod==0]
#print(atomnvt.values)
atom.loc[:,'x'] = atom.loc[:,'x']/0.529177
atom.loc[:,'y'] = atom.loc[:,'y']/0.529177
atom.loc[:,'z'] = atom.loc[:,'z']/0.529177
u = exatomic.Universe(atom=atom)
print("slice atom and vel and make uni--- %s seconds ---" % (time.time() - start_time))

del atom

u.atom.loc[:,'label'] = u.atom.get_atom_labels()
u.atom.loc[:,'label'] = u.atom['label'].astype(int)
u.frame.loc[:,'time'] = u.frame.index*dt*mod

vel.loc[:,'label'] = u.atom.label.values


# Add the unit cell dimensions to the frame table of the universe
a = celldm[pres]/0.529177
for i, q in enumerate(("x", "y", "z")):
    for j, r in enumerate(("i", "j", "k")):
        if i == j:
            u.frame[q+r] = a
        else:
            u.frame[q+r] = 0.0
    u.frame["o"+q] = 0.0
u.frame.loc[:,'periodic'] = True
print("add labels,time,celldm to uni--- %s seconds ---" % (time.time() - start_time))
#print(u.atom.tail().values)


u.compute_atom_two(vector=True,bond_extra=0.9)
print("compute_atom_two--- %s seconds ---" % (time.time() - start_time))
#print(u.atom_two.tail().values)

u.compute_molecule()
print("compute_molecule--- %s seconds ---" % (time.time() - start_time))
vel.loc[:,'molecule'] = u.atom.molecule.values

u.atom.sort_values(by=["molecule","label"],inplace=True)
u.atom.loc[:,'molecule_label']=u.atom[u.atom['frame']==u.atom.iloc[0]['frame']].molecule.values.tolist()*len(u.atom.frame.unique())
vel.sort_values(by=["molecule","label"],inplace=True)
vel.loc[:,'molecule_label']=u.atom.molecule_label.values
u.atom = u.atom[u.atom['molecule_label']<nmol]
vel = vel[vel['molecule_label']<nmol]
#print(u.atom.tail().values)
#print(vel.tail().values)

#print(u.atom[u.atom['molecule']==0].values)
#print(u.atom[u.atom['molecule']==0].label.values)
u.atom.loc[:,'mol-atom_index']=u.atom[u.atom['molecule']==0].label.values.tolist()*nmol*len(u.atom.frame.unique())
u.atom_two.loc[:,'molecule_label0'] = u.atom_two.atom0.map(u.atom['molecule_label'])
u.atom_two.loc[:,'molecule_label1'] = u.atom_two.atom1.map(u.atom['molecule_label'])
u.atom_two = u.atom_two[(u.atom_two['molecule_label0']<nmol) & (u.atom_two['molecule_label1']<nmol)]

u.atom.loc[:,'mass'] = u.atom.get_element_masses().values
vel.loc[:,'mass'] = u.atom.mass.values
print("sort,slice,and add masses to dfs--- %s seconds ---" % (time.time() - start_time))
#print(u.atom.tail().values)
#print(u.atom_two.tail().values)
#print(vel.tail().values)
gc.collect()

#vel = u.atom.copy()
#vel.loc[:,['x','y','z']] = u.atom.groupby('label')[['x','y','z']].apply(pd.DataFrame.diff)
#vel.loc[:,['x','y','z']] = vel.loc[:,['x','y','z']]/(u.atom.frame.diff().unique()[-1]*dt)

u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule']).astype(int)
u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule']).astype(int)
u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame']).astype(int)
u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
u.atom_two.loc[:,'atom_label0'] = u.atom_two.atom0.map(u.atom['label']).astype(int)
u.atom_two.loc[:,'atom_label1'] = u.atom_two.atom1.map(u.atom['label']).astype(int)
u.atom_two.loc[:,'mol-atom_index0'] = u.atom_two.atom0.map(u.atom['mol-atom_index']).astype(int)
u.atom_two.loc[:,'mol-atom_index1'] = u.atom_two.atom1.map(u.atom['mol-atom_index']).astype(int)
#print("Traj parsed....")
#print(u.atom_two.values)
print("map columns to atom_two--- %s seconds ---" % (time.time() - start_time))

u.atom['frame'] = u.atom['frame'].astype(int)
bonds = u.atom_two[u.atom_two['molecule0'] == u.atom_two['molecule1']]
del u.atom_two
vel['frame'] = vel['frame'].astype(int)
#vel = vel[vel['frame']>=start+mod]
#pos.to_csv(scratch+"pos.csv")
#bonds.to_csv(scratch+"bonds.csv")
vel.loc[:,'x'] = vel.loc[:,'x']/0.529177
vel.loc[:,'y'] = vel.loc[:,'y']/0.529177
vel.loc[:,'z'] = vel.loc[:,'z']/0.529177
#print(u.atom_two.tail().values)
#print(vel.tail().values)

pos_grouped = u.atom.groupby('molecule',observed=True)
del u.atom
bonds_grouped = bonds.groupby('molecule0',observed=True)
del bonds
vel_grouped = vel.groupby('molecule',observed=True)
del vel
#print(bonds_grouped.groups)
#pos.to_csv(scratch+'atom.csv')
#vel.to_csv(scratch+'vel.csv')
print("group dfs by mol--- %s seconds ---" % (time.time() - start_time))

#rot_mat = np.array([[-0.89910939, -0.18112621, -0.39849288],
#                    [ 0.09170104,  0.81223078, -0.57608322],
#                    [ 0.428012  , -0.554504  , -0.713674  ]])
if __name__=="__main__":
    mol_ax,av_ax, J = applyParallel3(SR_func1,pos_grouped,vel_grouped,bonds_grouped,mol_type=mol_type)
print("parallel compute angular vel,mom --- %s seconds ---" % (time.time() - start_time))
gc.collect()
#J = J.assign(frame=pos.loc[::5,'frame'].values,molecule=pos.loc[::5,'molecule'].values,molecule_label=pos.loc[::5,'molecule_label'].values)
#av.to_csv(scratch+'ang_vel_cart.csv')
out_prefix=temp+'-'+pres+'-test-'+traj+'-'
mol_ax.to_csv(path+out_prefix+'molax.csv',sep = ' ')
av_ax.to_csv(path+out_prefix+'ang_vel_molax.csv')
J.to_csv(path+out_prefix+'J_cart.csv')
print("write ax,vel,mom data--- %s seconds ---" % (time.time() - start_time))

J_acfs = applyParallel(correlate,J.groupby('molecule_label'),columns_in=['x','y','z'],columns_out=['$J_{x}$','$J_{y}$','$J_{z}$'],pass_columns=['frame','molecule_label','molecule'])
print("parallel compute acfs--- %s seconds ---" % (time.time() - start_time))
J_acfs.to_csv(path+out_prefix+'Jacfs_all.csv')
Jacf_mean=J_acfs.groupby('frame').apply(np.mean)
Jacf_mean['time']=Jacf_mean['frame']*dt*mod

Jacf_mean.to_csv(path+out_prefix+'Jacf.csv')
print("write acf data--- %s seconds ---" % (time.time() - start_time))
