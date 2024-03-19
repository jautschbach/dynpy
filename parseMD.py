import pandas as pd
import numpy as np
import scipy as sp
import os
#import exa
#import exatomic
#exa.logging.disable(level=10)
#exa.logging.disable(level=20)
#from exatomic import qe
import sys
import signal
from helper import which_trajs
from universe import Universe, Atom, Frame, Molecule, compute_frame_from_atom

def PARSE_MD(PD):
    try:
        parse_vel = PD.parse_vel
    except AttributeError:
        parse_vel = False
    try:
        start_prod = PD.start_prod
    except AttributeError:
        start_prod = None
    try:
        end_prod = PD.end_prod
    except AttributeError:
        end_prod = None
    
    user_time = which_trajs(PD)
    
    if PD.MD_ENGINE == "QE":
        try:
            symbols = PD.symbols
        except AttributeError:
            print("Missing required input variable symbols in class ParseDynamics for parsing QE. See dynpy_params.py")
            sys.exit(2)
        for i,traj in enumerate(PD.trajs):
            u,vel = parse_qe_md(traj,symbols,PD.sample_freq, start_prod, end_prod, parse_vel)
            us[i] = u
            vels[i] = vel
      
    elif PD.MD_ENGINE == "CP2K":
        try:
            md_print_freq = PD.md_print_freq
        except AttributeError:
            print("Missing required input variable md_print_freq in class ParseDynamics for parsing CP2K. See dynpy_params.py")
            sys.exit(2)
        for i,traj in enumerate(PD.trajs):
            u,vel = parse_cp2k_md(traj, PD.sample_freq, md_print_freq, start_prod, end_prod, parse_vel)
            us[i] = u
            vels[i] = vel
    
    
    elif PD.MD_ENGINE == "Tinker":
        try:
            md_print_freq = PD.md_print_freq
        except AttributeError:
            print("Missing required input variable md_print_freq in class ParseDynamics for parsing Tinker. See dynpy_params.py")
            sys.exit(2)
        try:
            nat = PD.nat
        except AttributeError:
            print("Missing required input variable nat in class ParseDynamics for parsing Tinker. See dynpy_params.py")
            sys.exit(2)
        us = {}
        vels = {}
        for i,traj in enumerate(PD.trajs):
            traj_dir = PD.traj_dir+traj
            u,vel = parse_tinker_md(traj_dir,PD.sample_freq, md_print_freq, nat, start_prod, end_prod, parse_vel)
            us[i] = u
            vels[i] = vel            
        #vel.to_csv("./vel.csv")
        #if parse_vel:
        #    vel = parse_tinker_vel(traj_dir,PD.sample_freq, md_print_freq, nat, start_prod, end_prod)
    #elif PD.MD_ENGINE == "prepared":
    #    u, vel = _prepared(pd.read_csv(traj_dir+"methane-01-atom-table.csv")
    
    else:
        print("MD_ENGINE not provided or not known. Implemented engines are QE, CP2K, and Tinker. Do you need to parse MD trajectories?")
        sys.exit(2)

    return us,vels

def parse_qe_md(traj_dir,symbols,sample_freq,start_prod=None,end_prod=None,parse_vel=False):
    try:
        pos = list(filter(lambda x: ".pos" in x, os.listdir(traj_dir)))[0]
    except:
        print("Did not find .pos trajectory file at " + traj_dir + " for parsing QE dynamics. Is this what you wanted?")
        sys.exit(2)
    atom = qe.parse_xyz(traj_dir+'/'+pos,symbols=symbols)
    atom['frame'] = atom['frame'].astype(int)
    if start_prod == None:
        start_prod = atom['frame'].iloc[0]
    if end_prod == None:
        end_prod = atom['frame'].iloc[-1]
    atom = atom[(atom['frame'] >= start_prod) & (atom['frame'] <= end_prod) & ((atom['frame']-start_prod)%sample_freq==0)]
    u = exatomic.Universe()
    u.atom = atom
    
    vel=None
    if parse_vel:
        vel_file = list(filter(lambda x: ".vel" in x, os.listdir(traj_dir)))[0]
        vel = qe.parse_xyz(traj_dir+'/'+vel_file,symbols=symbols)
        vel['frame'] = vel['frame'].astype(int)
        vel = vel[(vel['frame'] >= start_prod) & (vel['frame'] <= end_prod) & ((vel['frame']-start_prod)%sample_freq==0)]
    
    return u,vel

def parse_cp2k_md(traj_dir, sample_freq, md_print_freq, start_prod=None, end_prod=None, parse_vel=False):
    print("Reading trajectory output from CP2K...")
    pos = list(filter(lambda x: "pos" in x, os.listdir(traj_dir)))[0]
    xyz = exatomic.XYZ.from_file(traj_dir+'/'+pos)
    
    u = exatomic.XYZ.to_universe(xyz)
    u.atom['label'] = u.atom.get_atom_labels()
    u.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
    u.atom.loc[:,'frame'] = u.atom['frame'].astype(int)
    u.atom.loc[:,'frame'] = u.atom['frame']*md_print_freq
    if start_prod == None:
        start_prod = u.atom['frame'].iloc[0]
    if end_prod == None:
        end_prod = u.atom['frame'].iloc[-1]
    u.atom = u.atom[(u.atom['frame'] >= start_prod) & (u.atom['frame'] <= end_prod) & ((u.atom['frame']-start_prod)%sample_freq==0)]
    
    vel=None
    if parse_vel:
        vel_file = list(filter(lambda x: "vel" in x, os.listdir(traj_dir)))[0]
        velxyz = exatomic.XYZ.from_file(traj_dir+'/'+vel_file)
        velu = exatomic.XYZ.to_universe(velxyz)
        velu.atom['label'] = velu.atom.get_atom_labels()
        velu.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
        velu.atom.loc[:,'frame'] = velu.atom['frame'].astype(int)
        velu.atom.loc[:,'frame'] = velu.atom['frame']*md_print_freq
        velu.atom = velu.atom[(velu.atom['frame'] >= start_prod) & (velu.atom['frame'] <= end_prod) & ((velu.atom['frame']-start_prod)%sample_freq==0)]
    
    return u,velu.atom

def parse_tinker_md(traj_dir, sample_freq, md_print_freq, nat, start_prod=None, end_prod=None, parse_vel=True):
    arc = list(filter(lambda x: "arc" in x, os.listdir(traj_dir)))[0]
    cols = ['symbol','x','y','z']
    #read from .arc and eliminate comment lines
    atom = pd.read_csv(traj_dir+'/'+arc, sep=r'\s+', usecols=[1,2,3,4],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<(start_prod-1)*(nat+2))  |  (x%(nat+2)==0) | (x%(nat+2)==1) | ((x//(nat+2)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)//sample_freq+1))
    atom.loc[:,'symbol']=atom.loc[:,'symbol'].apply(normsym)
    atom.loc[:,'frame']=atom.index//nat
    #print(atom.head())
    atom.loc[:,'x'] = atom.loc[:,'x'].apply(d_to_e)/0.529177
    atom.loc[:,'y'] = atom.loc[:,'y'].apply(d_to_e)/0.529177
    atom.loc[:,'z'] = atom.loc[:,'z'].apply(d_to_e)/0.529177
    
    u = Universe(Atom(atom))
    #u.atom = atom

    u.atom.loc[:,'label'] = u.atom.get_atom_labels()
    u.atom.loc[:,'label'] = u.atom['label'].astype(int)
    #u.atom.loc[:,['x','y','z']] = u.atom[['x','y','z']].astype(float)
    #print(u.atom.head())
    vel=pd.DataFrame()
    if parse_vel:
        vel_file = list(filter(lambda x: "vel" in x, os.listdir(traj_dir)))[0]
        cols = ['symbol','x','y','z']
        #read from .vel and eliminate comment lines
        velatom = pd.read_csv(traj_dir+'/'+vel_file, sep=r'\s+', usecols=[1,2,3,4],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<(start_prod-1)*(nat+1))  |  (x%(nat+1)==0) | (x%(nat+1)==1) | ((x//(nat+1)-start_prod+1)%sample_freq!=0),dtype={'symbol':'category','x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)//sample_freq+1))
        velatom.loc[:,'symbol']=velatom.loc[:,'symbol'].apply(normsym)
        velatom.loc[:,'frame']=velatom.index//nat
        velatom.loc[:,'x'] = velatom.loc[:,'x'].apply(d_to_e)/0.529177
        velatom.loc[:,'y'] = velatom.loc[:,'y'].apply(d_to_e)/0.529177
        velatom.loc[:,'z'] = velatom.loc[:,'z'].apply(d_to_e)/0.529177
        velu = Universe(Atom=Atom(velatom))
        #velu.atom = velatom
        velu.atom.loc[:,'label'] = velu.atom.get_atom_labels()
        velu.atom.loc[:,'label'] = velu.atom['label'].astype(int)       
        vel = velu.atom
        #vel.to_csv("vel.csv")
    return u, vel

def _prepared(atom_data,timestep,start_prod,end_prod,celldm):
    u = Universe(atom_data)
    u.atom.loc[:,'x'] = u.atom.loc[:,'x']/0.529177
    u.atom.loc[:,'y'] = u.atom.loc[:,'y']/0.529177
    u.atom.loc[:,'z'] = u.atom.loc[:,'z']/0.529177
    u.atom.loc[:,'label'] = u.atom.get_atom_labels()
    u.atom.loc[:,'label'] = u.atom['label'].astype(int)
    u.atom.loc[:,'frame'] = u.atom['frame'].astype(int)
    
    u.atom = u.atom[(u.atom['frame'] >= start_prod) & (u.atom['frame'] <= end_prod)]

    vel = pd.DataFrame()    
    #vel.loc[:,['x','y','z']] = u.atom.groupby('label',group_keys=False)[['x','y','z']].apply(pd.DataFrame.diff)
    #vel.loc[:,['x','y','z']] = vel.loc[:,['x','y','z']]/(u.atom.frame.diff().unique()[-1]*timestep)
    #vel = vel.dropna(how='any')

    # Add the unit cell dimensions to the frame table of the universe
    u.frame = compute_frame_from_atom(u.atom)
    u.frame.add_cell_dm(celldm)
    u.compute_unit_atom()

    #u.compute_atom_two(vector=True,bond_extra=0.9)
    return u, vel

# def parse_tinker_vel(traj_dir,nat,start,end,mod):
#     vel_dat = list(filter(lambda x: "vel" in x, os.listdir(traj_dir)))[0]
#     cols = ['symbol','x','y','z']
#     #read from .arc and eliminate comment lines
#     vel = pd.read_csv(traj_dir+vel_dat,delim_whitespace=True,usecols=[1,2,3,4],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<(start-1)*(nat+1)) | (x%(nat+1)==0) | ((x//(nat+1)-start+1)%mod!=0) ,converters={'x':d_to_e,'y':d_to_e,'z':d_to_e},dtype={'symbol':'category'},nrows=nat*((end-start+1)//mod+1))
#     vel.loc[:,'symbol']=vel.loc[:,'symbol'].apply(normsym)
#     vel.loc[:,'frame']=vel.index//nat
#     vel.loc[:,'x'] = vel.loc[:,'x']/0.529177
#     vel.loc[:,'y'] = vel.loc[:,'y']/0.529177
#     vel.loc[:,'z'] = vel.loc[:,'z']/0.529177
    
#     return vel

def normsym(sym):
    sym=str(sym)
    if len(sym) > 1:
       if(sym[1].isupper()):
          sym = sym[0]
    return sym

def d_to_e(val):
    if 'D' in val:
        return float(val.replace('D', 'E'))
    else:
        return float(val)
