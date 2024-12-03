import pandas as pd
import string
#import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumulative_trapezoid
from numba import vectorize, jit
import os
import sys
#import seaborn as sns
#rc = {'legend.frameon': True, 'legend.fancybox': True, 'patch.facecolor': 'white', 'patch.edgecolor': 'black',
#       'axes.formatter.useoffset': False, 'text.usetex': True, 'font.weight': 'bold', 'mathtext.fontset': 'stix'}
#sns.set(context='poster', style='white', font_scale=1.7, font='serif', rc=rc)
#sns.set_style("ticks")
# import exa
# import exatomic
# from exatomic import qe
#import notebook as nb
import math
#import signal
#from dynpy import signal_handler
from nuc import *
nuc_df = pd.DataFrame.from_dict(nuc)
#signal.signal(signal.SIGINT, signal_handler)

def QR_module_main(QR,label=None):
    rawdf = read_efg_data(QR.data_set,dtype={'system':'category','traj':'i8','frame':'i8','time':'f8','label':'i8','symbol':'category','Vxx':'f8','Vxy':'f8','Vxz':'f8','Vyx':'f8','Vyy':'f8','Vyz':'f8','Vzx':'f8','Vzy':'f8','Vzz':'f8','V11':'f8','V22':'f8','V33':'f8','eta':'f8'})
    symbol = "".join([char for char in QR.analyte if char.isalpha()])
    
    #if QR.multiple_trajectories == False:
    #    rawdf['traj'] = 1
    #if QR.index_is_frame == True:
    #    rawdf['frame'] = rawdf.index.values
    #    rawdf['time'] = rawdf['frame']*QR.timestep
    #    rawdf['label'] = 1 
    #    #print("".join([c for c in QR.analyte if c.isalpha()]))
    #    rawdf['symbol'] = "".join([c for c in QR.analyte if c.isalpha()])
    #    rawdf['symbol'] = rawdf['symbol'].astype('category')
    ACFs = {}
    res = {}
    grouped = rawdf.groupby('traj')
    for traj, df in grouped:
        if traj == 0:
            continue
        #print(df.head().values)
        df = df.sort_values(by='time')
        df['frame'] = df['frame'] - df.frame.iloc[0]
        df['time'] = df['time'] - df.time.iloc[0]

        if label:
            #print(label)
            label = int(label)
            adf = df.groupby('label').get_group(label)
        else:
            adf = df.groupby('symbol', observed=True).get_group(symbol)

        spatial = cart_to_spatial(adf,pass_columns=['traj','system'])
        acfs = spatial.groupby('label').apply(correlate, pass_columns=['system','symbol'])
        #print(acfs)
        #print(acfs.dtypes)
        acf_mean = acfs[['frame','time','$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$']].groupby('frame').apply(np.mean,axis=0)
        acf_mean.loc[:,'symbol'] = symbol

        #print(acf_mean)
        #print(acf_mean.dtypes)
        ACFs[traj] = acf_mean

        g = spectral_dens(acf_mean, cutoff=False, cutoff_tol=1e-3)
        #print(g.values)
        v = normal_factor(acf_mean)
        t = correlation_time(g,v)
        rax = relaxation(g,QR.analyte)
        rax['$\\tau_{c}$'] = t['$\\tau_{c}$']
        rax[r'$\langle V(0)^2\rangle$'] = v[r'$\langle V(0)^2\rangle$']

        res[traj] = rax

    rax = pd.concat(res)
    rax.index = rax.index.droplevel(level=1)
    #print(rax)
    rax_mean = pd.DataFrame(rax.mean(numeric_only=True)).T
    rax_mean.index = ["mean"]
    #print(rax_mean)
    rax_sem = pd.DataFrame(rax.sem(numeric_only=True)).T
    rax_sem.index = ["err"]
    #print(rax_sem)
    all_rax = pd.concat([rax,rax_mean,rax_sem],sort=False)
    acfs = pd.concat(ACFs)
    system = QR.data_set.split('.csv')[0]
    all_rax.to_csv(system+"-"+symbol+"-relax.csv",index_label="traj")
    acfs.to_csv(system+"-"+symbol+"-acfs.csv",index=False)
    print("Results written to "+system+"-"+symbol+"-relax.csv")
    return(all_rax, acfs)
    print("Done")

def read_efg_data(file_path,ensemble_average=False,dtype=None):
    rawdf = pd.read_csv(file_path,dtype=dtype)
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
    for column in pass_columns:
        if column in cartdf.columns:
            spatial[column] = cartdf[column]

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
    acf[['frame','time','label']] = df[['frame','time','label']].astype(float)
    for column in pass_columns:
        if column in df.columns:
            acf[column] = df[column]#.astype('category')
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

        g = acf.iloc[:bounds[l]][columns_in].apply(sp.integrate.simpson,x=acf.iloc[:bounds[l]]['time'])
        #g = pd.DataFrame([np.trapz(acf.iloc[:bounds[l]][m], x=acf.iloc[:bounds[l]]['time']) for l,m in enumerate(columns_in)]).transpose()
        #g.columns = columns_out
        #g['$g_{iso}$'] = np.mean([g[m] for m in columns_out])
        #g['symbol'] = acf['symbol'].iloc[0]


    else:
        if dt:
            g = acf[columns_in].apply(sp.integrate.simpson, dx=dt)
        else:
            g = acf[columns_in].apply(sp.integrate.simpson, x=acf['time'])

    g.index = columns_out
    g['$g_{iso}$'] = g.mean()
    g['symbol'] = acf['symbol'].iloc[0]

    return g

def normal_factor(acf,columns_in=['$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$'],columns_out=[r'$\sigma_{2,-2}$', r'$\sigma_{2,-1}$', r'$\sigma_{2,0}$', r'$\sigma_{2,1}$', r'$\sigma_{2,2}$']):
    V = acf[columns_in].iloc[0]
    V.index = columns_out
    V[r'$\langle V(0)^2\rangle$'] = V.values.sum()
    V['symbol'] = acf['symbol'].iloc[0]
    #print(V)
    return V

def correlation_time(spec_dens,norm):
    #print(norm.values)
    tau = pd.DataFrame.from_dict({r'$\tau_{2,'+str(m)+'}$':[spec_dens['$g_{2,'+str(m)+'}$']/norm[r'$\sigma_{2,'+str(m)+'}$']] for m in range(-2,3)})
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


