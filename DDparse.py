import pandas as pd
import string
import numpy as np
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumtrapz
import math
#import dask
from ddrax import *
from parseMD import *
import gc
import time
import parseMD
from helper import which_trajs

def DD_module_main(PD,DD):
       outer_start_time = time.time()

       try:
            analyte_label=DD.analyte_label
       except AttributeError:
            analyte_label = None

       try:
            analyte_species = DD.analyte_species
       except AttributeError:
            analyte_species = '1H'
       
       try:
            pair_type = DD.pair_type
            if pair_type.casefold() not in ['intra','inter','all']:
                   pair_type = 'all'
                   print("Inter- and intramolecular pairs will be considered.\n")
       except AttributeError:
            pair_type = 'all'
       
       user_time = which_trajs(PD)
       res = {}
       for t,traj in enumerate(PD.trajs):
            inner_start_time = time.time()
            u,vel = parseMD.PARSE_MD(PD,traj)
              
            #traj = str(t).zfill(2)
            #print(u.atom.head())
            print("computing two-atom distances...")
            u = prep_DD_uni1(u,PD,DD)
            #print(u.atom)
            time1 = time.time()
            print("--- {t:.2f} seconds ---".format(t = time1 - inner_start_time))
            gc.collect()
             
            if analyte_label:
                   XH = u.atom_two[(u.atom_two['mol-atom_index0']==analyte_label) & (u.atom_two['symbol1']=='H')]
            else:
                   XH = u.atom_two[(u.atom_two['symbol0']=='H') & (u.atom_two['symbol1']=='H')]
            if pair_type == 'intra':
                   XH = XH[XH['molecule0'] == XH['molecule1']]
            elif pair_type == 'inter':
                   XH = XH[XH['molecule0'] != XH['molecule1']]
             #print("writing dists...:")
             #XH.to_csv(PD.traj_dir+"test_XH.csv")
             #intraHH = pd.read_csv('/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/01/NVE/TIP3P/water-01-gas-intraHH.csv',
             #                 usecols=['label0','label1','frame','time',
             #                          'dx','dy','dz','dr','molecule0','molecule1'],
             #                 dtype={'label0':'category','label1':'category','frame':'i8',
             #                        'time':'f8','dx':'f8','dy':'f8','dz':'f8',
             #                        'dr':'f8','molecule0':'i8','molecule1':'i8'})
             #allHH = allHH[allHH['time']!='time']                        
             #allHH = allHH.astype({'frame':'i8','time':'f8','dx':'f8','dy':'f8','dz':'f8','dr':'f8','molecule0':'i8','molecule1':'i8'})
             #allHH = allHH[(allHH['frame']>20)]                                            
             #XH.sort_values(by='frame',inplace=True)
             #interHH = allHH[allHH['molecule0']!=allHH['molecule1']]
             #intraHH = allHH[allHH['molecule0']==allHH['molecule1']]
             #alldd = cart_to_dipolar_from_df(allHH)
             #alldd.to_csv('test.csv')
             #del allHH
             #interdd = cart_to_dipolar_from_df(interHH)
             #del interHH
            print("computing dipolar quantities...")
            dd = cart_to_dipolar_from_df(XH)
            #dd.to_csv(PD.traj_dir+'test-DD.csv')
            del XH
            gc.collect()
            time2 = time.time()
            print("--- {t:.2f} seconds ---".format(t = time2 - time1))
            #acf = alldd.reset_index().groupby(['label0','label1']).apply(correlate)
            #del alldd
            #interacf = interdd.reset_index().groupby(['label0','label1']).apply(correlate)
            #del interdd
            print("computing TCFs...")
            acfs = dd.reset_index().groupby(['label0','label1']).apply(correlate)
            
            del dd
            gc.collect()
            time3 = time.time()
            print("--- {t:.2f} seconds ---".format(t = time3 - time2))
            N = len(pd.unique(acfs[['label0','label1']].values.ravel('K')))
            #interN = len(pd.unique(interacf[['label0','label1']].values.ravel('K')))
            #intraN = len(pd.unique(intraacf[['label0','label1']].values.ravel('K')))

            #acf_avg = ((acf.groupby('time')['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$', '$G_{r}$','$G_{2,0}$','$G_{2,1}$','$G_{2,2}$'].sum().dropna())*2/N).reset_index()
            #del acf
            acf_avg = ((acfs.groupby('time')[['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$', '$G_{r}$','$G_{2,0}$','$G_{2,1}$','$G_{2,2}$']].sum().dropna())*2/N).reset_index()
            #del interacf
            #acf_avg = acfs.groupby('time')[['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$', '$G_{r}$','$G_{2,0}$','$G_{2,1}$','$G_{2,2}$']].apply(np.mean,axis=0).reset_index()
              
            del acfs
            gc.collect()

            #EN = spec_dens_extr_narr(acf_avg)
            acf_avg.to_csv(PD.traj_dir+traj+'/avg-acf.csv')
            #del acf_avg
            #interEN = spec_dens_extr_narr(interacf_avg)
            #interacf_avg.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-interacf.csv')
            #del interacf_avg
            EN = spec_dens_extr_narr(acf_avg)
            #intraacf_avg.to_csv('/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/01/NVE/TIP3P/DD-intra-ACF.csv')
            #print(EN)
            #fullrax = dipolar(EN, 'H')
            #interrax = dipolar(interEN, 'H')
            rax,tc,F = dipolar(EN, symbol1=analyte_species,symbol2='1H')
            print("1/T1/nH = {r:.3e} Hz, tau_c = {t:.3e} ps, <F(0)^2> = {f:.3e} au^-6".format(r=rax,t=tc,f=F))
            print("For intramolecular contrib. in heteronuclear systems (e.g. C--H relaxation of 13C), multiply 1/T1/nH by number of bonded spins to obtain 1/T1. E.g. for C-13 in methyl group, nH=3")
            res[traj] = [rax,tc,F]
            #rax = fullrax.join(intrarax,lsuffix='',rsuffix='intra')
            #fullrax.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-totalrax.csv')
            #interrax.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-interrax.csv')
            #intrarax.to_csv('/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/01/NVE/TIP3P/DD-intra-rax.csv')
       res_df = pd.DataFrame(res).T
       res_df.columns=["1/T1/nH","tau_c","<F(0)^2>"]
       res_df.to_csv(PD.traj_dir+'DD-results.csv')
def prep_DD_uni1(u,PD,DD):
    try:
          dmax = DD.rmax
    except AttributeError:
          dmax = 15   
    u.atom = Atom(u.atom)
    # Add the unit cell dimensions to the frame table of the universe
    u.frame = compute_frame_from_atom(u.atom)
    u.frame.add_cell_dm(celldm = PD.celldm)
    u.compute_unit_atom()

    u.atom['label'] = u.atom.get_atom_labels()
    #u.atom[['x','y','z']] = u.atom[['x','y','z']].astype(float)
    u.atom.frame = u.atom.frame.astype(int)

    u.compute_atom_two(dmax=dmax,vector=True,bond_extra=0.45)
    u.compute_molecule()
    u.molecule.classify((DD.identifier,'analyte',True))
    #u.atom = u.atom[u.atom['molecule_label'] < nmol]
    nat_per_mol  = np.sum([int(n) for n in DD.identifier.replace('(',')').split(')') if n.isnumeric()])
    mol_atom_labels = [n for n in range(nat_per_mol)]
    u.atom.loc[:,'mol-atom_index']=mol_atom_labels*(len(u.atom.frame)//len(mol_atom_labels))
    #u.atom_two.loc[:,'molecule_label0'] = u.atom_two.atom0.map(u.atom['molecule_label'])
    #u.atom_two.loc[:,'molecule_label1'] = u.atom_two.atom1.map(u.atom['molecule_label'])
    #u.atom_two = u.atom_two[(u.atom_two['molecule_label0'] < DD.nmol) & (u.atom_two['molecule_label1'] < DD.nmol)]
    u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule']).astype(int)
    u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule']).astype(int)
    u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame']).astype(int)
    u.atom_two.loc[:,'time'] = u.atom_two['frame']*PD.timestep
    u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
    u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
    u.atom_two.loc[:,'label0'] = u.atom_two.atom0.map(u.atom['label']).astype(int)
    u.atom_two.loc[:,'label1'] = u.atom_two.atom1.map(u.atom['label']).astype(int)
    u.atom_two.loc[:,'mol-atom_index0'] = u.atom_two.atom0.map(u.atom['mol-atom_index']).astype(int)
    u.atom_two.loc[:,'mol-atom_index1'] = u.atom_two.atom1.map(u.atom['mol-atom_index']).astype(int)

    return u
