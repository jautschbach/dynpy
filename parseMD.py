import pandas as pd
import numpy as np
import scipy as sp
import os
import itertools
import sys
import signal

from universe import Universe, Atom, Frame, Molecule, compute_frame_from_atom

def PARSE_MD(PD,traj):
    if 'parse_vel' not in PD.__dict__.keys():
        PD.parse_vel = False
    if 'sample_freq' not in PD.__dict__.keys():
        PD.sample_freq = 1

    if PD.MD_format == "QE":
        if 'symbols' not in PD.__dict__.keys():
            print("Error: Missing required input variable symbols in class ParseDynamics for parsing QE.")
            sys.exit(2)
        #for i,traj in enumerate(PD.trajs):
        u,vel = parse_qe_md(traj_dir=traj, symbols=PD.symbols, sample_freq=PD.sample_freq, celldm=PD.celldm, start_prod=PD.start_prod, end_prod=PD.end_prod,parse_vel=PD.parse_vel)
            #us[i] = u
            #vels[i] = vel
      
    #elif PD.MD_format == "CP2K":
    #    try:
    #        md_print_freq = PD.md_print_freq
    #    except AttributeError:
    #        print("Missing required input variable md_print_freq in class ParseDynamics for parsing CP2K. See dynpy_params.py")
    #        sys.exit(2)
        #for i,traj in enumerate(PD.trajs):
        #u,vel = parse_cp2k_md(traj, PD.sample_freq, md_print_freq, start_prod, end_prod)
            #us[i] = u
            #vels[i] = vel
    
    
    elif PD.MD_format == "Tinker":
        #print("Tinker")
        #us = {}
        #vels = {}
        #for i,traj in enumerate(PD.trajs):
        traj_dir = PD.traj_dir+traj
        u,vel = parse_tinker_md(traj_dir,sample_freq=PD.sample_freq, md_print_freq=PD.md_print_freq, nat=PD.nat, start_prod=PD.start_prod, end_prod=PD.end_prod, parse_vel=PD.parse_vel)
            #us[i] = u
            #vels[i] = vel            
        #vel.to_csv("./vel.csv")
        #if parse_vel:
        #    vel = parse_tinker_vel(traj_dir,PD.sample_freq, md_print_freq, nat, start_prod, end_prod)
    #elif PD.MD_ENGINE == "prepared":
    #    u, vel = _prepared(pd.read_csv(traj_dir+"methane-01-atom-table.csv")
    elif PD.MD_format == "xyz":
        #us = {}
        #vels = {}
        #print(PD.traj_dir+PD.trajs[0],PD.sample_freq, md_print_freq, nat, start_prod, end_prod)
        #for i,traj in enumerate(PD.trajs):
        traj_dir = PD.traj_dir+traj
        u,vel = parse_xyz(traj_dir, sample_freq=PD.sample_freq, md_print_freq=PD.md_print_freq, nat=PD.nat, start_prod=PD.start_prod, end_prod=PD.end_prod,parse_vel=PD.parse_vel)
            #us[i] = u
            #vels[i] = vel
    else:
        print("MD_format not provided or not recognized. Implemented formats are 'QE', 'Tinker', and 'xyz'. Do you need to parse MD trajectories?")
        sys.exit(2)
    #print(us[0].atom)
    return u,vel

def parse_qe_md(traj_dir,symbols,sample_freq,celldm,start_prod,end_prod,parse_vel=False):
    try:
        pos = list(filter(lambda x: ".pos" in x, os.listdir(traj_dir)))[0]
    except:
        print("Did not find .pos trajectory file at " + traj_dir + " for parsing QE dynamics. Is this what you wanted?")
        sys.exit(2)
    #if start_prod == None:
    #    start_prod = atom['frame'].iloc[0]
    #if end_prod == None:
    #    end_prod = atom['frame'].iloc[-1]
    
    nat = len(symbols)
    cols = ['x','y','z']
    #read from .xyz and eliminate comment lines
    atom = pd.read_csv(traj_dir+'/'+pos, sep=r'\s+', usecols=[0,1,2],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<=(start_prod-1)*(nat+1))  |  (x%(nat+1)==0) | ((x//(nat+1)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)+sample_freq)/sample_freq)
    with open(traj_dir+'/'+pos, 'r') as f:
        lines = f.readlines()
    frames = np.empty((end_prod-start_prod+sample_freq)//sample_freq)
    f = 0
    for i,line in enumerate(lines):
        if ((i>=(start_prod-1)*(nat+1))  &  (i%(nat+1)==0) & (i<=(end_prod)*(nat+1)) & ((i//(nat+1)-start_prod+1)%sample_freq==0)):
            #print(i)
            #print(line.split())
            frames[f] = line.split()[0]
            f += 1
    fframes = [[frame]*nat for frame in frames]
    full_frames = list(itertools.chain.from_iterable(fframes))
    atom.loc[:,'symbol']=symbols*(len(atom)//nat)
    atom.loc[:,'frame']= full_frames #atom.index//nat
    #print(atom.tail())
    atom.loc[:,'x'] = atom.loc[:,'x']
    atom.loc[:,'y'] = atom.loc[:,'y']
    atom.loc[:,'z'] = atom.loc[:,'z']
    #nat_frames = [nat]*len(atom.frame.unique())
    #frame = pd.DataFrame(nat_frames,index = atom.frame.unique(),columns=['atom_count'])
    
    u = Universe(Atom(atom))
    #u.atom = atom

    u.atom.loc[:,'label'] = u.atom.get_atom_labels()
    u.atom.loc[:,'label'] = u.atom['label'].astype(int)
    u.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
    
    # Add the unit cell dimensions to the frame table of the universe
    u.frame = compute_frame_from_atom(u.atom)
    u.frame.add_cell_dm(celldm = celldm)
    u.compute_unit_atom()
    #for i, q in enumerate(("x", "y", "z")):
    #    for j, r in enumerate(("i", "j", "k")):
    #        if i == j:
    #            frame[q+r] = celldm
    #        else:
    #            frame[q+r] = 0.0
    #    frame["o"+q] = 0.0
    #frame['periodic'] = True

    #u.atom.loc[:,['x','y','z']] = u.atom[['x','y','z']].astype(float)
    #print(u.atom.tail())
    vel=pd.DataFrame()
    if parse_vel:
        vel_file = list(filter(lambda x: ".vel" in x, os.listdir(traj_dir)))[0]
        cols = ['x','y','z']
        #read from .vel and eliminate comment lines
        velatom = pd.read_csv(traj_dir+'/'+vel_file, sep=r'\s+', usecols=[1,2,3],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<=(start_prod-1)*(nat+1))  |  (x%(nat+1)==0) | ((x//(nat+1)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)+sample_freq)/sample_freq)
        velatom.loc[:,'symbol']=symbols
        velatom.loc[:,'frame']=full_frames
        velatom.loc[:,'x'] = velatom.loc[:,'x']
        velatom.loc[:,'y'] = velatom.loc[:,'y']
        velatom.loc[:,'z'] = velatom.loc[:,'z']
        velu = Universe(Atom(velatom))
        #velu.atom = velatom
        velu.atom.loc[:,'label'] = velu.atom.get_atom_labels()
        velu.atom.loc[:,'label'] = velu.atom['label'].astype(int)
        velu.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
        vel = velu.atom
        #vel.to_csv("vel.csv")
    #print(u.atom.tail())
    #print(vel.tail())
    return u, vel


# def parse_cp2k_md(traj_dir, sample_freq, md_print_freq, start_prod=None, end_prod=None, parse_vel=False):
#     print("Reading trajectory output from CP2K...")
#     pos = list(filter(lambda x: "pos" in x, os.listdir(traj_dir)))[0]
#     xyz = XYZ.from_file(traj_dir+'/'+pos)
    
#     u = XYZ.to_universe(xyz)
#     u.atom['label'] = u.atom.get_atom_labels()
#     u.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
#     u.atom.loc[:,'frame'] = u.atom['frame'].astype(int)
#     u.atom.loc[:,'frame'] = u.atom['frame']*md_print_freq
#     if start_prod == None:
#         start_prod = u.atom['frame'].iloc[0]
#     if end_prod == None:
#         end_prod = u.atom['frame'].iloc[-1]
#     u.atom = u.atom[(u.atom['frame'] >= start_prod) & (u.atom['frame'] <= end_prod) & ((u.atom['frame']-start_prod)%sample_freq==0)]
    
#     vel=None
#     if parse_vel:
#         vel_file = list(filter(lambda x: "vel" in x, os.listdir(traj_dir)))[0]
#         velxyz = exatomic.XYZ.from_file(traj_dir+'/'+vel_file)
#         velu = exatomic.XYZ.to_universe(velxyz)
#         velu.atom['label'] = velu.atom.get_atom_labels()
#         velu.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
#         velu.atom.loc[:,'frame'] = velu.atom['frame'].astype(int)
#         velu.atom.loc[:,'frame'] = velu.atom['frame']*md_print_freq
#         velu.atom = velu.atom[(velu.atom['frame'] >= start_prod) & (velu.atom['frame'] <= end_prod) & ((velu.atom['frame']-start_prod)%sample_freq==0)]
    
#     return u,velu.atom

def parse_tinker_md(traj_dir, sample_freq, md_print_freq, nat, start_prod, end_prod, parse_vel=False):
    try:
        arc = list(filter(lambda x: "arc" in x, os.listdir(traj_dir)))[0]
    except:
        print("Did not find .arc trajectory file at " + traj_dir + " for parsing Tinker dynamics. Is this what you wanted?")
        sys.exit(2)
    print("reading "+ arc)
    cols = ['symbol','x','y','z']
    #read from .arc and eliminate comment lines
    atom = pd.read_csv(traj_dir+'/'+arc, sep=r'\s+', usecols=[1,2,3,4],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<=(start_prod-1)*(nat+2))  |  (x%(nat+2)==0) | (x%(nat+2)==1) | ((x//(nat+2)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)+sample_freq)/sample_freq)
    atom.loc[:,'symbol']=atom.loc[:,'symbol'].apply(normsym)
    #print(atom)
    frames = np.arange(start_prod,end_prod+sample_freq,sample_freq)
    frames *= md_print_freq
    #print(frames)
    fframes = [[frame]*nat for frame in frames]
    full_frames = list(itertools.chain.from_iterable(fframes))
    atom.loc[:,'frame']=full_frames
    #print(atom.head())
    atom.loc[:,'x'] = atom.loc[:,'x'].apply(d_to_e)/0.529177
    atom.loc[:,'y'] = atom.loc[:,'y'].apply(d_to_e)/0.529177
    atom.loc[:,'z'] = atom.loc[:,'z'].apply(d_to_e)/0.529177
    
    u = Universe(Atom(atom))
    #u.atom = atom

    u.atom.loc[:,'label'] = u.atom.get_atom_labels()
    u.atom.loc[:,'label'] = u.atom['label'].astype(int)
    u.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
    #u.atom.loc[:,['x','y','z']] = u.atom[['x','y','z']].astype(float)
    #print(u.atom.head())
    vel=pd.DataFrame()
    if parse_vel:
        vel_file = list(filter(lambda x: ".vel" in x, os.listdir(traj_dir)))[0]
        #print(vel_file)
        cols = ['symbol','x','y','z']
        #read from .vel and eliminate comment lines
        velatom = pd.read_csv(traj_dir+'/'+vel_file, sep=r'\s+', usecols=[1,2,3,4],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<=(start_prod-1)*(nat+2))  |  (x%(nat+2)==0) | (x%(nat+2)==1) | ((x//(nat+2)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)+sample_freq)/sample_freq)
        #print(velatom.head())
        velatom.loc[:,'symbol']=velatom.loc[:,'symbol'].apply(normsym)
        velatom.loc[:,'frame']=full_frames
        velatom.loc[:,'x'] = velatom.loc[:,'x'].apply(d_to_e)/0.529177
        velatom.loc[:,'y'] = velatom.loc[:,'y'].apply(d_to_e)/0.529177
        velatom.loc[:,'z'] = velatom.loc[:,'z'].apply(d_to_e)/0.529177
        velu = Universe(Atom(velatom))
        #velu.atom = velatom
        velu.atom.loc[:,'label'] = velu.atom.get_atom_labels()
        velu.atom.loc[:,'label'] = velu.atom['label'].astype(int)
        velu.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)   
        vel = velu.atom
        #vel.to_csv("vel.csv")
    return u, vel

def parse_xyz(traj_dir, sample_freq, md_print_freq, nat, start_prod=None, end_prod=None, parse_vel=True):
    try:
        xyz = list(filter(lambda x: ".xyz" in x, os.listdir(traj_dir)))[0]
    except:
        print("Did not find .xyz trajectory file at " + traj_dir + " for parsing dynamics. Is this what you wanted?")
        sys.exit(2)
    cols = ['symbol','x','y','z']
    #read from .xyz and eliminate comment lines
    atom = pd.read_csv(traj_dir+'/'+xyz, sep=r'\s+', usecols=[0,1,2,3],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<=(start_prod-1)*(nat+2))  |  (x%(nat+2)==0) | (x%(nat+2)==1) | ((x//(nat+2)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)+sample_freq)/sample_freq)
    atom.loc[:,'symbol']=atom.loc[:,'symbol'].apply(normsym)
    frames = np.arange(start_prod,end_prod+sample_freq,sample_freq)
    frames *= md_print_freq
    fframes = [[frame]*nat for frame in frames]
    full_frames = list(itertools.chain.from_iterable(fframes))
    atom.loc[:,'frame']=full_frames
    #print(atom.tail())
    atom.loc[:,'x'] = atom.loc[:,'x'].apply(d_to_e)/0.529177
    atom.loc[:,'y'] = atom.loc[:,'y'].apply(d_to_e)/0.529177
    atom.loc[:,'z'] = atom.loc[:,'z'].apply(d_to_e)/0.529177
    
    u = Universe(Atom(atom))
    #u.atom = atom

    u.atom.loc[:,'label'] = u.atom.get_atom_labels()
    u.atom.loc[:,'label'] = u.atom['label'].astype(int)
    u.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
    #u.atom.loc[:,['x','y','z']] = u.atom[['x','y','z']].astype(float)
    #print(u.atom.tail())
    vel=pd.DataFrame()
    if parse_vel:
        vel_file = list(filter(lambda x: ".vel" in x, os.listdir(traj_dir)))[0]
        cols = ['symbol','x','y','z']
        #read from .vel and eliminate comment lines
        velatom = pd.read_csv(traj_dir+'/'+vel_file, sep=r'\s+', usecols=[0,1,2,3],names=cols,header=None,na_filter=False,skiprows=lambda x: (x<=(start_prod-1)*(nat+2))  |  (x%(nat+2)==0) | (x%(nat+2)==1) | ((x//(nat+2)-start_prod+1)%sample_freq!=0),dtype={'symbol':str,'x':str,'y':str,'z':str},nrows=nat*((end_prod-start_prod)+sample_freq)/sample_freq)
        velatom.loc[:,'symbol']=velatom.loc[:,'symbol'].apply(normsym)
        velatom.loc[:,'frame']=full_frames
        velatom.loc[:,'x'] = velatom.loc[:,'x'].apply(d_to_e)/0.529177
        velatom.loc[:,'y'] = velatom.loc[:,'y'].apply(d_to_e)/0.529177
        velatom.loc[:,'z'] = velatom.loc[:,'z'].apply(d_to_e)/0.529177
        velu = Universe(Atom(velatom))
        #velu.atom = velatom
        velu.atom.loc[:,'label'] = velu.atom.get_atom_labels()
        velu.atom.loc[:,'label'] = velu.atom['label'].astype(int)
        velu.atom.drop_duplicates(['frame','label'], keep='last',inplace=True)
        vel = velu.atom
        #vel.to_csv("vel.csv")
    #print(u.atom.tail())
    #print(vel.tail())
    return u, vel

def _prepared(atom_data,timestep,start_prod,end_prod,celldm,units='Angstrom'):
    u = Universe(atom_data)
    if units=='Angstrom':
        u.atom.loc[:,'x'] = u.atom.loc[:,'x']/0.529177
        u.atom.loc[:,'y'] = u.atom.loc[:,'y']/0.529177
        u.atom.loc[:,'z'] = u.atom.loc[:,'z']/0.529177
    u.atom.loc[:,'label'] = u.atom.get_atom_labels()
    u.atom.loc[:,'label'] = u.atom['label'].astype(int)
    u.atom.loc[:,'frame'] = u.atom['frame'].astype(int)
    
    #u.atom = u.atom[(u.atom['frame'] >= start_prod) & (u.atom['frame'] <= end_prod)]

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
       if((sym[1].isupper()) | sym[1].isnumeric()):
          sym = sym[0]
    return sym

def d_to_e(val):
    if 'D' in val:
        return float(val.replace('D', 'E'))
    else:
        return float(val)
