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
exa.logging.disable(level=10)
exa.logging.disable(level=20)
from exatomic import qe
import math
from parseMD import *
#from neighbors_input import *
from SRrax import *
import gc
from multiprocessing import Pool, cpu_count, set_start_method
#import ray
import time

def SpinRotation(dynpy_params):
    start_time = time.time()
    PD = dynpy_params.ParseDynamics
    SR = dynpy_params.SpinRotation

    trajs = parse_many(PD)
    for traj in trajs:
        pos_dir = PD.traj_dir + traj + '/'
        u, vel = PARSE_MD(pos_dir, PD)
        u.atom['label'] = u.atom.get_atom_labels()
        #print(u.atom.head())
        #vel.to_csv("./vel.csv")
        if not vel: # Estimate velocities from atoms and timestep
            print("Explicit velocities not provided. Will be determined from position and timestep. If timestep is too large, these velocities will be inaccurate.")
            u.atom.frame = u.atom.frame.astype(int)
            vel = u.atom.copy()
            vel.loc[:,['x','y','z']] = u.atom.groupby('label',group_keys=False)[['x','y','z']].apply(pd.DataFrame.diff)
            vel.loc[:,['x','y','z']] = vel.loc[:,['x','y','z']]/(u.atom.frame.diff().unique()[-1]*PD.timestep)
            vel = vel.dropna(how='any')
            u.atom = u.atom[u.atom['frame'] > PD.start_prod]
        #u.atom.to_csv("./atom.csv")
        #vel.to_csv("./vel.csv")

        u.compute_atom_two(vector=True,bond_extra=0.9)
        time1 = time.time()
        print("compute_atom_two                        --- %s seconds ---" % (time1 - start_time))
        
        u.compute_molecule()
        time2 = time.time()
        print("compute_molecule                        --- %s seconds ---" % (time2 - time1))
        #print(u.molecule) 
        if SR.mol_type == 'water':
            u.molecule.classify(('H(2)O(1)','water',True))
        elif SR.mol_type == 'acetonitrile':
            u.molecule.classify(('H(3)C(2)N(1)','acetonitrile',True))
        elif SR.mol_type == 'methane':
            u.molecule.classify(('H(4)C(1)','methane',True))
        #print(u.molecule.head())

        u.atom.loc[:,'classification'] = u.atom.molecule.map(u.molecule.classification)
        #print(u.atom.head())
        u.atom = u.atom[u.atom['classification'] == SR.mol_type]
        
        labels = pd.DataFrame(u.atom.groupby('molecule').apply(lambda x: tuple(x['label'])),columns=['mol_atom_labels'])
        molecule_labels = labels.groupby('mol_atom_labels').nunique().reset_index().reset_index().rename(columns={'index':'molecule_label'}).set_index('mol_atom_labels')
        labels['molecule_label'] = labels.mol_atom_labels.map(molecule_labels.molecule_label)
        u.atom['molecule_label'] = u.atom.molecule.map(labels.molecule_label)
        sets = u.atom.groupby('frame').apply(lambda x: set(list(x['molecule_label']))).values.tolist()
        contiguos_molecules = set.intersection(*sets)
        u.atom = u.atom[u.atom['molecule_label'].isin(contiguos_molecules)]

        u.atom.loc[:,'molecule'] = u.atom.loc[:,'molecule'].values.astype(int)
        u.atom.sort_values(by=["molecule","label"],inplace=True)
        #u.atom.loc[:,'molecule_label']=list(range(len(u.atom[u.atom['frame']==u.atom.iloc[0]['frame']].molecule.values)))*len(u.atom.frame.unique())
        #u.atom.loc[:,'molecule_label']=u.atom[u.atom['frame']==u.atom.iloc[0]['frame']].molecule.values.tolist()*len(u.atom.frame.unique())
        u.atom = u.atom[u.atom['molecule_label']<SR.nmol]

        #print(u.atom.head())
        #print(len(u.atom[u.atom['molecule']==0].label.values.tolist()*SR.nmol*len(u.atom.frame.unique())))
        mol_atom_labels = [n for n in range(len(u.atom[u.atom['molecule']==0]))]
        u.atom.loc[:,'mol-atom_index']=mol_atom_labels*(len(u.atom.frame)//len(mol_atom_labels))
        #print(u.atom.head())
        #print(u.atom.tail())
        #u.atom.loc[:,'mol-atom_index']=mol_atom_labels*SR.nmol*len(u.atom.frame.unique())
        u.atom_two.loc[:,'molecule_label0'] = u.atom_two.atom0.map(u.atom['molecule_label'])
        u.atom_two.loc[:,'molecule_label1'] = u.atom_two.atom1.map(u.atom['molecule_label'])
        u.atom_two = u.atom_two[(u.atom_two['molecule_label0']<SR.nmol) & (u.atom_two['molecule_label1']<SR.nmol)]

        u.atom.loc[:,'mass'] = u.atom.get_element_masses().values
        
        time3 = time.time()
        print("sort,slice,and add masses to dfs        --- %s seconds ---" % (time3 - time2))

        gc.collect()
        
        #u.atom.to_csv("./atom.csv")
        #vel.to_csv("./vel.csv")
        #u.atom_two.to_csv("./atom_two.csv")
        vel['molecule'] = vel.index.map(u.atom['molecule'])
        vel.dropna(how='any',inplace=True)
        vel.sort_values(by=["molecule","label"],inplace=True)
        vel.loc[:,'molecule_label']=u.atom.molecule_label.values
        vel = vel[vel['molecule_label']<SR.nmol]
        vel.loc[:,'mass'] = u.atom.mass.values
            
        
        u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule']).astype(int)
        u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule']).astype(int)
        u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame']).astype(int)
        u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
        u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
        u.atom_two.loc[:,'atom_label0'] = u.atom_two.atom0.map(u.atom['label']).astype(int)
        u.atom_two.loc[:,'atom_label1'] = u.atom_two.atom1.map(u.atom['label']).astype(int)
        u.atom_two.loc[:,'mol-atom_index0'] = u.atom_two.atom0.map(u.atom['mol-atom_index']).astype(int)
        u.atom_two.loc[:,'mol-atom_index1'] = u.atom_two.atom1.map(u.atom['mol-atom_index']).astype(int)

        time4 = time.time()
        print("map columns to atom_two                 --- %s seconds ---" % (time4 - time3))

        u.atom['frame'] = u.atom['frame'].astype(int)
        bonds = u.atom_two[u.atom_two['molecule0'] == u.atom_two['molecule1']]
        del u.atom_two
        vel['frame'] = vel['frame'].astype(int)

        pos_grouped = u.atom.groupby('molecule',observed=True)
        #del u.atom
        bonds_grouped = bonds.groupby('molecule0',observed=True)
        #del bonds
        vel_grouped = vel.groupby('molecule',observed=True)
        #del vel
        #print(bonds_grouped.groups)
        #pos.to_csv(scratch+'atom.csv')
        #vel.to_csv(scratch+'vel.csv')
        time5 = time.time()
        print("group dataframes by molecule            --- %s seconds ---" % (time5 - time4))

        #rot_mat = np.array([[-0.89910939, -0.18112621, -0.39849288],
        #                    [ 0.09170104,  0.81223078, -0.57608322],
        #                    [ 0.428012  , -0.554504  , -0.713674  ]])
        #if __name__=="__main__":
        #print(pos_grouped.head(6))
        #print(vel_grouped.head(6))
        #print(bonds_grouped.head())
        mol_ax,av_ax, J = applyParallel3(SR_func1,pos_grouped,vel_grouped,bonds_grouped,mol_type=SR.mol_type)
        mol_ax.to_csv(pos_dir+'mol_ax.csv')
        av_ax.to_csv(pos_dir+'av_ax.csv')
        J.to_csv(pos_dir+'J.csv')
        #else:
        #    print("__name__ != main SRparse.py")
        #    sys.exit(2)

        time6 = time.time()
        print("parallel compute angular vel,momentum   --- %s seconds ---" % (time6 - time5))
        gc.collect()
        #J = J.assign(frame=pos.loc[::5,'frame'].values,molecule=pos.loc[::5,'molecule'].values,molecule_label=pos.loc[::5,'molecule_label'].values)
        #av.to_csv(scratch+'ang_vel_cart.csv')
        #out_prefix=temp+'-'+pres+'-test-'+traj+'-'
        #mol_ax.to_csv(path+out_prefix+'molax.csv',sep = ' ')
        #av_ax.to_csv(path+out_prefix+'ang_vel_molax.csv')
        #J.to_csv(path+out_prefix+'J_cart.csv')
        #print("write ax,vel,mom data--- %s seconds ---" % (time.time() - start_time))

        J_acfs = applyParallel(correlate,J.groupby('molecule_label'),columns_in=['x','y','z'],columns_out=['$J_{x}$','$J_{y}$','$J_{z}$'],pass_columns=['frame','molecule_label','molecule'])
        time7 = time.time()
        print("parallel compute acfs                   --- %s seconds ---" % (time7 - time6))
        J_acfs.to_csv(pos_dir+'Jacfs_all.csv')
        Jacf_mean=J_acfs.groupby('frame').apply(np.mean, axis=0)
        Jacf_mean['time']=Jacf_mean['frame']*PD.timestep

        #Jacf_mean.to_csv(path+out_prefix+'Jacf.csv')
        #print("write acf data--- %s seconds ---" % (time.time() - start_time))

        C = [float(c)*1000*2*np.pi for c in SR.C_SR]
        #C_par = C_1
        #C_perp = (C_2+C_3)/2
        #c_a = 1/3*(2*C_perp+C_par)
        #c_d = C_perp-C_par
        G = spec_dens(Jacf_mean,columns_in=['$J_{x}$','$J_{y}$','$J_{z}$'])
        #t1 = G['$G_3$']/acf.loc[0,'$G_3$']
        #t2 = (G['$G_1$']/acf.loc[0,'$G_1$'] + G['$G_2$']/acf.loc[0,'$G_2$'])/2
        #v1 = acf.loc[0,'$G_3$']#/41341.375**2
        #v2 = (acf.loc[0,'$G_1$'] + acf.loc[0,'$G_2$'])/2#/41341.375**2
        r = 2/3/(sp.constants.hbar**2)*(G[0]*C[0]**2 + G[1]*C[1]**2 + G[2]*C[2]**2) * (1e12)*(5.29177e-11)**4 * (1.66054e-27)**2
        #rax[traj] = (t1,v1,t2,v2,r)
        print("1/T1     =     %s Hz" % r)
        print("Total SR Run Time     --- %s seconds ---" % (time.time() - start_time))
