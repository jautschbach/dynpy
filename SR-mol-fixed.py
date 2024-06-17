import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumtrapz
from numba import vectorize, jit
import os
import sys
import exa
import exatomic
exa.logging.disable(level=10)
exa.logging.disable(level=20)
from exatomic import qe
import math
from SRrax import *
import gc
import signal as sig
from dynpy import signal_handler

#sig.signal(sig.SIGINT, signal_handler)

traj = 'ss'
scratch = "/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/"+traj+'/NVE/TIP3P/'
atom = pd.read_csv(scratch+"water-"+traj+"-NVE-TIP3P-2000ps-atom-table.csv")
dt = 0.1
start = 50
end = 20050
mod = 10
nframes = (end-start)//mod
nmol = 100

atom['frame'] = atom['frame'].astype(int)
atomnvt = atom[(atom['frame']>=start) & (atom['frame']<=end) & (atom['frame']%mod==0)]
atomnvt.loc[:,'x'] = atomnvt.loc[:,'x']/0.529177
atomnvt.loc[:,'y'] = atomnvt.loc[:,'y']/0.529177
atomnvt.loc[:,'z'] = atomnvt.loc[:,'z']/0.529177
u = exatomic.Universe(atom=atomnvt)

u.atom.loc[:,'label'] = u.atom.get_atom_labels()
u.atom.loc[:,'label'] = u.atom['label'].astype(int)
u.atom = u.atom[u.atom['label']<3*nmol]

u.frame.loc[:,'time'] = u.frame.index*dt

# Add the unit cell dimensions to the frame table of the universe
a = 274.75/0.529177
for i, q in enumerate(("x", "y", "z")):
    for j, r in enumerate(("i", "j", "k")):
        if i == j:
            u.frame[q+r] = a
        else:
            u.frame[q+r] = 0.0
    u.frame["o"+q] = 0.0
u.frame.loc[:,'periodic'] = True


u.compute_atom_two(vector=True,bond_extra=0.9)

u.compute_molecule()


u.atom.loc[:,'molecule_label']=u.atom[u.atom['frame']==u.atom.iloc[0]['frame']].molecule.values.tolist()*len(u.atom.frame.unique())

u.atom.loc[:,'mass'] = u.atom.get_element_masses().values

del atom
del atomnvt
gc.collect()

u.atom = u.atom.groupby('molecule').apply(rel_CoM)
vel = u.atom.copy()
vel.loc[:,['x','y','z','frame']] = u.atom.groupby('label')[['x','y','z','frame']].apply(pd.DataFrame.diff)
vel.loc[:,['x','y','z']] = vel.loc[:,['x','y','z']]/(vel.frame.unique()[-1]*dt)

#check that all molecules are full waters
#lens = []
#for mol,v in u.atom.groupby('molecule'):
#    if len(v) != 3:
#        print(mol,len(v))

Rs = u.atom.groupby('molecule').apply(make_R)
Is = u.atom.groupby('molecule').apply(make_I)
gc.collect()

u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule'])
u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule'])
u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame'])
u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
u.atom_two.loc[:,'label0'] = u.atom_two.atom0.map(u.atom['label'])
u.atom_two.loc[:,'label1'] = u.atom_two.atom1.map(u.atom['label'])

pos_grouped = u.atom.groupby('molecule')
vel_grouped = vel.groupby('molecule')
bonds_grouped = u.atom_two.groupby(['molecule0','molecule1'])


omegas = np.empty(((nframes*nmol),),dtype=[('1','f8'),('2','f8'),('3','f8'),('molecule','i8'),('frame','i8')])
i=0
for mol,pos in pos_grouped:
    if pos.frame.iloc[0]>=start+mod:
        #if i==0:
        #    print("***DEBUG****:  "+str(pos.frame.iloc[0]))
        rv = cross(pos,vel_grouped.get_group(mol))
        o = la.solve(Rs[mol],rv)
        ax = water_fixed_coord(bonds_grouped.get_group((mol,mol)))
        #os[i] = (o[0],o[1],o[2],mol,pos.frame.iloc[0])
        oI = np.matmul(ax,o)
        I_ax = np.matmul(ax,np.matmul(Is[mol],la.inv(ax)))
        #oIs[i] = (oI[0],oI[1],oI[2],mol,pos.frame.iloc[0])
        omega = np.matmul(I_ax.T,oI)
        omegas[i] = (omega[0],omega[1],omega[2],mol,pos.frame.iloc[0])
        i+=1


Omegas = pd.DataFrame(omegas)
#Omegas.to_csv(scratch+"test.csv")
#print(str(len(Omegas)))
Omegas.loc[:,'molecule_label'] = list(range(0,nmol))*nframes

acfs = Omegas.groupby('molecule_label').apply(correlate,columns_in = ['1','2','3'],columns_out=['$G_1$','$G_2$','$G_3$'],pass_columns=['molecule_label','frame'])

acf_avg = acfs.groupby('frame').apply(np.mean)
acf_avg.loc[:,'$G_{avg}$'] = acf_avg[['$G_1$','$G_2$','$G_3$']].apply(np.mean,axis=1)
acf_avg.loc[:,'time'] = acf_avg.loc[:,'frame']*dt
acf_avg[['int_1','int_2','int_3']] = np.cumsum(acf_avg[["$G_1$","$G_2$",'$G_3$']])

#C_1=16.495*2*np.pi*1000
#C_2=16.495*2*np.pi*1000
#C_3=-1.875*2*np.pi*1000
#G = spec_dens(acf_avg,columns_in=['$G_1$','$G_2$','$G_3$','$G_{avg}$'])

#r = 2/3/(sp.constants.hbar**2)*(G[0]*C_1**2 + G[1]*C_2**2 + G[2]*C_3**2) * (1e12)*(5.29177e-11)**4 * (1.66054e-27)**2

acf_avg.to_csv(scratch+'water-'+traj+'-NVE-TIP3P-mol-fixed-2000ps-acf_avg.csv')
#with open(scratch+'methane-'+traj+'-mol-fixed.rax','w') as f:
#    f.write(str(r))
