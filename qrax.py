import pandas as pd
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumtrapz
from numba import vectorize, jit
import os
import sys
import seaborn as sns
rc = {'legend.frameon': True, 'legend.fancybox': True, 'patch.facecolor': 'white', 'patch.edgecolor': 'black',
       'axes.formatter.useoffset': False, 'text.usetex': True, 'font.weight': 'bold', 'mathtext.fontset': 'stix'}
sns.set(context='poster', style='white', font_scale=1.7, font='serif', rc=rc)
sns.set_style("ticks")
import exa
import exatomic
from exatomic import qe
import notebook as nb
import math
from nuc import *
nuc_df = pd.DataFrame.from_dict(nuc)

def read_efg_data(file_path,ensemble_average=False):
    rawdf = pd.io.parsers.read_csv(file_path)
    #if rawdf.isnull().any().any():
    #    print("WARNING: Missing data in "+file_path+". Will be interpolated for computation of correlation functions")
    #if 'label' not in rawdf.columns:
    #    rawdf['label']=1
    #if 'traj' not in rawdf.columns:
    #    rawdf['traj']='01'
    #if ensemble_average:
    #    grouped = rawdf.groupby('label')
    #    #dfs = [d[1].reset_index(drop=True) for d in grouped]
    #    #dfss = []', n=2*N)
    #    return grouped
    #else:
    return rawdf

#@vectorize(nopython=True)
def r20(v):
    return 3*np.sqrt(1/6)*v['Vzz']

def r2_1(v):
    return complex(v['Vxz'],-v['Vyz'])

def r21(v):
    return complex(-v['Vxz'], -v['Vyz'])

def r2_2(v):
    a = (1/2)*(v['Vxx'] - v['Vyy'])
    return complex(a,-v['Vxy'])

def r22(v):
    a = (1/2)*(v['Vxx'] - v['Vyy'])
    return complex(a,v['Vxy'])

def cart_to_spatial(cartdf,pass_columns):
    try:
        time = cartdf['time']
    except KeyError:
        time=None
    spatial = pd.DataFrame.from_dict({"frame":cartdf['frame'], "time":time,"symbol":cartdf['symbol'],
                                   "label":cartdf['label'], "$R_{2,-2}$":cartdf.apply(r2_2, axis=1),
                                   "$R_{2,-1}$":cartdf.apply(r2_1, axis=1), "$R_{2,0}$":cartdf.apply(r20, axis=1),
                                   "$R_{2,1}$":cartdf.apply(r21, axis=1), "$R_{2,2}$":cartdf.apply(r22, axis=1)})
    spatial[pass_columns] = cartdf[pass_columns]

    return spatial

def wiener_khinchin(f):
    #“Wiener-Khinchin theorem”
    real = pd.Series(np.real(f)).interpolate()
    imag = pd.Series(np.imag(f)).interpolate()
    f = pd.Series([complex(r,i) for r,i in zip(real,imag)])
    N = len(f)
    fvi = np.fft.fft(f,n=2*N)
    acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
    acf = acf/N
    return acf

def correlate(df, pass_columns, columns_in=['$R_{2,-2}$', '$R_{2,-1}$', '$R_{2,0}$', '$R_{2,1}$', '$R_{2,2}$'],
              columns_out=['$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$']):
    acf = df[columns_in].apply(wiener_khinchin)
    acf.columns = columns_out
    acf[['frame','time','label']] = df[['frame','time','label']]
    acf[pass_columns] = df[pass_columns]

    return acf


def spectral_dens(acf,dt=None,columns_in=['$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$'],
                  columns_out=['$g_{2,-2}$', '$g_{2,-1}$', '$g_{2,0}$', '$g_{2,1}$', '$g_{2,2}$'], cutoff=False, cutoff_tol=1e-3):
    #print(acf)
    if cutoff:
        bounds=[0,0,0,0,0]
        for l,m in enumerate(columns_in):
            for i,b in enumerate(np.isclose(pd.Series(acf[m]/acf.iloc[0][m]).rolling(window=100).mean(),0,atol=cutoff_tol)):
                if i == len(acf)-1:
                    bounds[l]=len(acf)-1
                else:
                    if b:
                        bounds[l]=i
                        break
        #print(bounds)

        g = acf.iloc[:bounds[l]][columns_in].apply(sp.integrate.simps,x=acf.iloc[:bounds[l]]['time'])
        #g = pd.DataFrame([np.trapz(acf.iloc[:bounds[l]][m], x=acf.iloc[:bounds[l]]['time']) for l,m in enumerate(columns_in)]).transpose()
        #g.columns = columns_out
        #g['$g_{iso}$'] = np.mean([g[m] for m in columns_out])
        #g['symbol'] = acf['symbol'].iloc[0]


    else:
        if dt:
            g = acf[columns_in].apply(sp.integrate.simps, dx=dt)
        else:
            g = acf[columns_in].apply(sp.integrate.simps, x=acf['time'])

    g.index = columns_out
    g['$g_{iso}$'] = g.mean()
    g['symbol'] = acf['symbol'].iloc[0]

    return g

def normal_factor(acf,columns_in=['$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$'],columns_out=['$\sigma_{2,-2}$', '$\sigma_{2,-1}$', '$\sigma_{2,0}$', '$\sigma_{2,1}$', '$\sigma_{2,2}$']):
    V = acf[columns_in].iloc[0]
    V.index = columns_out
    V[r'$\langle V(0)^2\rangle$'] = V.sum()
    V['symbol'] = acf['symbol'].iloc[0]

    return V

def correlation_time(spec_dens,norm):
    tau = pd.DataFrame.from_dict({r'$\tau_{2,'+str(m)+'}$':[spec_dens['$g_{2,'+str(m)+'}$']/norm['$\sigma_{2,'+str(m)+'}$']] for m in range(-2,3)})
    tau[r'$\tau_{c}$'] = spec_dens['$g_{iso}$']*5/norm[r'$\langle V(0)^2\rangle$']
    tau['symbol'] = spec_dens['symbol']
    return tau

def relaxation(spec_dens,analyte):
    quad_mom = nuc_df.loc['Q',analyte]
    s = nuc_df.loc['I',analyte]
    C_q = constants.e*quad_mom/constants.hbar
    s_const = (2*s+3)/(s**2*(2*s-1))
    au_q = 9.71736408e21

    g0 = 4*spec_dens['$g_{2,2}$'] + spec_dens['$g_{2,1}$'] + spec_dens['$g_{2,-1}$'] + 4*spec_dens['$g_{2,-2}$']
    g1 = 2*spec_dens['$g_{2,-2}$'] + 3*spec_dens['$g_{2,-1}$'] + 2*spec_dens['$g_{2,1}$'] + 3*spec_dens['$g_{2,0}$']
    g = 10*spec_dens['$g_{iso}$']

    longitudinal = (1/40)*C_q**2*s_const*g0*au_q**2*1e-12
    transverse = (1/40)*C_q**2*s_const*g1*au_q**2*1e-12
    isotropic = (1/40)*C_q**2*s_const*g*au_q**2*1e-12

    return pd.DataFrame.from_dict({'symbol':[spec_dens['symbol']], r'$\frac{1}{T_{1}}$':[longitudinal],r'$\frac{1}{T_{2}}$':[transverse], r'$\frac{1}{T_{iso}}$':[isotropic]})

def Q_rax(efg_data,analyte,label=None):
    rawdf = read_efg_data(efg_data)
    symbol = "".join([char for char in analyte if char.isalpha()])
    ACFs = {}
    res = {}

    rawdf[['time','frame','Vxx','Vxy','Vxz','Vyx','Vyy','Vyz','Vzx','Vzy','Vzz']]=rawdf[['time','frame','Vxx','Vxy','Vxz','Vyx','Vyy','Vyz','Vzx','Vzy','Vzz']].astype(float)
    for traj, df in rawdf.groupby('traj'):
        df = df.sort_values(by='time')
        df['frame'] = df['frame'] - df.frame.iloc[0]
        df['time'] = df['time'] - df.time.iloc[0]
        if label:
            adf = df.groupby('label').get_group(label)
        else:
            adf = df.groupby('symbol').get_group(symbol)

        spatial = cart_to_spatial(adf,pass_columns=['traj','system'])
        acfs = correlate(spatial, pass_columns=['traj','system','symbol'])
        ACFs[traj] = acfs

        g = spectral_dens(acfs, cutoff=False, cutoff_tol=1e-3)
        v = normal_factor(acfs)
        t = correlation_time(g,v)
        rax = relaxation(g,analyte)
        rax['$\\tau_{c}$'] = t['$\\tau_{c}$']
        rax['$\\langle V(0)^2\\rangle$'] = v['$\\langle V(0)^2\\rangle$']

        res[traj] = rax

    rax = pd.concat(res)
    rax.index = rax.index.droplevel(level=1)
    #print(rax)
    rax_mean = pd.DataFrame(rax.mean()).T
    rax_mean.index = ["mean"]
    #print(rax_mean)
    rax_sem = pd.DataFrame(rax.sem()).T
    rax_sem.index = ["err"]
    #print(rax_sem)
    all_rax = rax.append(rax_mean,sort=False).append(rax_sem,sort=False)
    acfs_all = pd.concat(ACFs)

    system = efg_data.split('-efg')[0]
    all_rax.to_csv(system+"-relax.csv",index_label="traj")
    print("Results written to "+system+"-relax.csv")
    return(all_rax, acfs_all)
    print("Done")
