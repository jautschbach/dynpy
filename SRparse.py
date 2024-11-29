import pandas as pd
import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy import signal
from scipy import constants
import os
import sys
#import exa
#import exatomic
#exa.logging.disable(level=10)
#exa.logging.disable(level=20)
#from exatomic import qe
import math
from parseMD import *
#from neighbors_input import *
from SRrax import *
import gc
from multiprocessing import Pool, cpu_count, set_start_method
#import ray
import time
import signal as sig
from helper import user_confirm, which_trajs

def SR_module_main(PD,SR):
    outer_start_time = time.time()
    res = {}
    for t,traj in enumerate(PD.trajs):
        inner_start_time = time.time()
        u,vel = PARSE_MD(PD,traj)
        u,vel = prep_SR_uni1(u,vel,PD,SR)
        time1 = time.time()
        print("compute_atom_two                           --- {t:.2f} seconds ---".format(t = time1 - inner_start_time))
        #print(u.atom.tail())
        u,vel = prep_SR_uni2(u,vel,PD,SR)
        time2 = time.time()
        print("compute, classify, and label molecules     --- {t:.2f} seconds ---".format(t = time2 - time1))

        pos_grouped, vel_grouped, bonds_grouped = prep_SR_uni3(u,vel)
        #pos_grouped.get_group(0).to_csv(PD.traj_dir+"pos_grouped.csv")
        #vel_grouped.get_group(0).to_csv(PD.traj_dir+"vel_grouped.csv")
        #bonds_grouped.get_group(0).to_csv(PD.traj_dir+"bonds_grouped.csv")
        time3 = time.time()
        print("group dataframes by molecule               --- {t:.2f} seconds ---".format(t = time3 - time2))

        try:
            methyl_indeces=SR.methyl_indeces
        except AttributeError:
            methyl_indeces = None
        try:
            mol_plane_indeces=SR.mol_plane_indeces
        except AttributeError:
            mol_plane_indeces = None
        try:
            global_momentum=SR.global_momentum
        except AttributeError:
            global_momentum = False
        #print(global_momentum)
        #print(SR.mol_type)
        #print(methyl_indeces)
        #print(global_momentum)
        mol_ax, av_ax, J = applyParallel3(SR_func1, pos_grouped, vel_grouped, bonds_grouped, mol_type=SR.mol_type, methyl_indeces=methyl_indeces,global_momentum=global_momentum)
        time4 = time.time()
        print("parallel compute angular vel,momentum      --- {t:.2f} seconds ---".format(t = time4 - time3))

        #rot_mat = np.array([[-0.89910939, -0.18112621, -0.39849288],
        #                    [ 0.09170104,  0.81223078, -0.57608322],
        #                    [ 0.428012  , -0.554504  , -0.713674  ]])
        #if __name__=="__main__":
        #print(pos_grouped.head(6))
        #print(vel_grouped.head(6))
        #print(bonds_grouped.head())

        #J = J.assign(frame=pos.loc[::5,'frame'].values,molecule=pos.loc[::5,'molecule'].values,molecule_label=pos.loc[::5,'molecule_label'].values)
        #av.to_csv(scratch+'ang_vel_cart.csv')
        #out_prefix=temp+'-'+pres+'-test-'+traj+'-'
        mol_ax.to_csv(PD.traj_dir+traj+'/molax.csv')
        av_ax.to_csv(PD.traj_dir+traj+'/ang_vel_molax.csv')
        J.to_csv(PD.traj_dir+traj+'/J_cart.csv')
        #print("write ax,vel,mom data--- %03.2s seconds ---" % (time.time() - start_time))
        
        J_acfs = applyParallel(correlate,J.groupby('molecule_label'),columns_in=['x','y','z'],columns_out=['$J_{x}$','$J_{y}$','$J_{z}$'],pass_columns=['frame','molecule_label','molecule'])
        Jacf_mean=J_acfs.groupby('frame').apply(np.mean, axis=0)
        Jacf_mean['time']=Jacf_mean['frame']*PD.timestep
        time5 = time.time()
        print("parallel compute acfs                      --- {t:.2f} seconds ---".format(t = time5 - time4))

        """Write data to csv?"""
        J_acfs.to_csv(PD.traj_dir+traj+'/Jacfs_all.csv')
        #mol_ax.to_csv(pos_dir+'mol_ax.csv')
        #av_ax.to_csv(pos_dir+'av_ax.csv')
        #J.to_csv(pos_dir+'J.csv')

        Jacf_mean.to_csv(PD.traj_dir+traj+'/Jacf.csv')
        #print("write acf data--- %03.2s seconds ---" % (time.time() - start_time))

        tx,ty,tz,r = compute_SR_rax(Jacf_mean,SR,PD)
        print("1/T1     =     {t:.4f} Hz".format(t=r))
        print("Total SR Run Time for {s}    --- {t:.2f} seconds ---".format(s = PD.traj_dir+traj, t = time.time() - inner_start_time))
        res[traj] = tx,ty,tz,r
        gc.collect()
    res_df = pd.DataFrame(res).T
    res_df.columns=["tau_x","tau_y","tau_z","1/T1"]
    res_df.to_csv(PD.traj_dir+'SR-results.csv')
    
    print("Total SR Run Time         --- {t:.2f} seconds ---".format(t = time.time() - outer_start_time))


def prep_SR_uni1(u,vel,PD,SR,p_vel=True):
    u.atom = Atom(u.atom)
    # Add the unit cell dimensions to the frame table of the universe
    u.frame = compute_frame_from_atom(u.atom)
    u.frame.add_cell_dm(celldm = PD.celldm)
    u.compute_unit_atom()

    u.atom['label'] = u.atom.get_atom_labels()
    #u.atom[['x','y','z']] = u.atom[['x','y','z']].astype(float)
    u.atom.frame = u.atom.frame.astype(int)
    #print(u.atom.frame.unique())
    #print(u.atom[u.atom['frame']>PD.start_prod])
    #vel.to_csv("./vel.csv")
    #print(u.atom.tail())
    try:
        parse_vel = PD.parse_vel
    except AttributeError:
        parse_vel = False
    if (vel.empty) and (parse_vel==False): # Estimate velocities from atoms and timestep
        print("Explicit velocities not provided. Will be determined from position and timestep. If timestep is too large, these velocities will be inaccurate.")
        u.atom.frame = u.atom.frame.astype(int)
        vel = u.atom.copy()
        vel.loc[:,['x','y','z']] = u.atom.groupby('label',group_keys=False,observed=False)[['x','y','z']].apply(pd.DataFrame.diff)
        vel.loc[:,['x','y','z']] = vel.loc[:,['x','y','z']]/(u.atom.frame.diff().unique()[-1]*PD.timestep)
        vel = vel.dropna(how='any')
        u.atom = u.atom[u.atom['frame'] > u.atom.iloc[0]['frame']]
    
    #u.atom = u.atom[((u.atom['frame']-PD.start_prod) % SR.sample_freq) == 0]
    #vel = vel[((vel['frame']-PD.start_prod) % SR.sample_freq) == 0]
    u.compute_atom_two(vector=True,bond_extra=0.45)
  
    return u, vel
        
def prep_SR_uni2(u, vel, PD, SR, pop_vel=True):
    #print(u.atom)
    u.compute_molecule()
    #print(u.molecule.head(20))
    if SR.mol_type == 'water':
        u.molecule.classify(('H(2)O(1)','water',True))
        nat_per_mol = 3
    elif SR.mol_type == 'acetonitrile':
        u.molecule.classify(('H(3)C(2)N(1)','acetonitrile',True))
        nat_per_mol = 6
    elif SR.mol_type == 'methane':
        u.molecule.classify(('H(4)C(1)','methane',True))
        nat_per_mol = 5
    elif SR.mol_type == 'methyl':
        u.molecule.classify((SR.identifier,'methyl',True))
        #print([int(n) for n in SR.identifier.replace('(',')').split(')') if n.isnumeric()])
        nat_per_mol  = np.sum([int(n) for n in SR.identifier.replace('(',')').split(')') if n.isnumeric()])
        #print(nat_per_mol)
    elif SR.mol_type == 'ring':
        u.molecule.classify((SR.identifier,'ring',True))
        #print([int(n) for n in SR.identifier.replace('(',')').split(')') if n.isnumeric()])
        nat_per_mol  = np.sum([int(n) for n in SR.identifier.replace('(',')').split(')') if n.isnumeric()])
        #print(nat_per_mol)
    u.atom.loc[:,'classification'] = u.atom.molecule.map(u.molecule.classification)
    
    u.atom = u.atom[u.atom['classification'] == SR.mol_type]
    
    labels = pd.DataFrame(u.atom.groupby('molecule',observed=False).apply(lambda x: tuple(x['label'])),columns=['mol_atom_labels'])
    labels = labels[labels['mol_atom_labels']!=()]
    molecule_labels = labels.groupby('mol_atom_labels').nunique().reset_index().reset_index().rename(columns={'index':'molecule_label'}).set_index('mol_atom_labels')
    #print(molecule_labels)
    labels['molecule_label'] = labels.mol_atom_labels.map(molecule_labels.molecule_label)
    #print(labels)
    u.atom['molecule_label'] = u.atom.molecule.map(labels.molecule_label).astype(int)
    #print(u.atom)
    sets = u.atom.groupby('frame').apply(lambda x: set(list(x['molecule_label']))).values.tolist()
    contiguos_molecules = set.intersection(*sets)
    u.atom = u.atom[u.atom['molecule_label'].isin(contiguos_molecules)]
    #print(u.atom.head())
    u.atom.loc[:,'molecule'] = u.atom.loc[:,'molecule'].values.astype(int)
    u.atom.sort_values(by=["molecule","label"],inplace=True)
    #u.atom.loc[:,'molecule_label']=list(range(len(u.atom[u.atom['frame']==u.atom.iloc[0]['frame']].molecule.values)))*len(u.atom.frame.unique())
    #u.atom.loc[:,'molecule_label']=u.atom[u.atom['frame']==u.atom.iloc[0]['frame']].molecule.values.tolist()*len(u.atom.frame.unique())
    u.atom = u.atom[u.atom['molecule_label'] < SR.nmol]

    #print(u.atom.head)
    #print(len(u.atom[u.atom['molecule']==0].label.values.tolist()*SR.nmol*len(u.atom.frame.unique())))
    mol_atom_labels = [n for n in range(nat_per_mol)]
    #print(mol_atom_labels)
    #print(len(u.atom.frame))
    u.atom.loc[:,'mol-atom_index']=mol_atom_labels*(len(u.atom.frame)//len(mol_atom_labels))
    #print(u.atom.head())
    #print(u.atom.tail())
    #u.atom.loc[:,'mol-atom_index']=mol_atom_labels*SR.nmol*len(u.atom.frame.unique())
    u.atom_two.loc[:,'molecule_label0'] = u.atom_two.atom0.map(u.atom['molecule_label'])
    u.atom_two.loc[:,'molecule_label1'] = u.atom_two.atom1.map(u.atom['molecule_label'])
    u.atom_two = u.atom_two[(u.atom_two['molecule_label0']<SR.nmol) & (u.atom_two['molecule_label1']<SR.nmol)]
    
    if pop_vel==True:
        vel.loc[:,'molecule'] = vel.index.map(u.atom['molecule'])
        vel.loc[:,'mol-atom_index'] = vel.index.map(u.atom['mol-atom_index'])
        vel.dropna(how='any',inplace=True)
        vel.sort_values(by=["molecule","label"],inplace=True)
        vel.loc[:,'molecule_label']=u.atom.molecule_label.values
        vel = vel[vel['molecule_label']<SR.nmol]
        vel.loc[:,'mass'] = u.atom.loc[:,'mass'].values
            
    u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule']).astype(int)
    u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule']).astype(int)
    u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame']).astype(int)
    u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
    u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
    u.atom_two.loc[:,'atom_label0'] = u.atom_two.atom0.map(u.atom['label']).astype(int)
    u.atom_two.loc[:,'atom_label1'] = u.atom_two.atom1.map(u.atom['label']).astype(int)
    u.atom_two.loc[:,'mol-atom_index0'] = u.atom_two.atom0.map(u.atom['mol-atom_index']).astype(int)
    u.atom_two.loc[:,'mol-atom_index1'] = u.atom_two.atom1.map(u.atom['mol-atom_index']).astype(int)

    return u, vel
    
def prep_SR_uni3(u,vel):
    u.atom['frame'] = u.atom['frame'].astype(int)
    bonds = u.atom_two[u.atom_two['molecule0'] == u.atom_two['molecule1']]
    del u.atom_two
    #print(vel)
    vel['frame'] = vel['frame'].astype(int)

    pos_grouped = u.atom.groupby('molecule',observed=True)
    #del u.atom
    bonds_grouped = bonds.groupby('molecule0',observed=True)
    #del bonds
    vel_grouped = vel.groupby('molecule',observed=True)
    #del vel
    #print(bonds_grouped.groups)
    #pos_grouped.to_csv(scratch+'atom.csv')
    #vel.to_csv(scratch+'vel.csv')

    return pos_grouped, vel_grouped, bonds_grouped
    
def compute_SR_rax(Jacf_mean,SR,PD):
    C = [float(c)*1000*2*np.pi for c in SR.C_SR]
    #C_par = C_1
    #C_perp = (C_2+C_3)/2
    #c_a = 1/3*(2*C_perp+C_par)
    #c_d = C_perp-C_par
    G = spec_dens(Jacf_mean,columns_in=['$J_{x}$','$J_{y}$','$J_{z}$'])
    #print(G)
    #print(Jacf_mean)
    tx = G.iloc[0]/Jacf_mean.iloc[0]['$J_{x}$']
    ty = G.iloc[1]/Jacf_mean.iloc[0]['$J_{y}$']
    tz = G.iloc[2]/Jacf_mean.iloc[0]['$J_{z}$']
    #v1 = acf.loc[0,'$G_3$']#/41341.375**2
    #v2 = (acf.loc[0,'$G_1$'] + acf.loc[0,'$G_2$'])/2#/41341.375**2
    r = 2/3/(sp.constants.hbar**2)*(G.iloc[0]*C[0]**2 + G.iloc[1]*C[1]**2 + G.iloc[2]*C[2]**2) * (1e12)*(5.29177e-11)**4 * (1.66054e-27)**2
    #rax[traj] = (t1,v1,t2,v2,r)
    #with open(PD.traj_dir+PD.prefix+"_SRrax.out",'w') as f:
    #    f.write("""spectral densities:\n
    #            g_x =  """+str(G[0])+"""\n
    #            g_y =  """+str(G[1])+"""\n
    #            g_z =  """+str(G[2])+"""\n
    #            R1 =   """+str(r))
    return tx,ty,tz,r