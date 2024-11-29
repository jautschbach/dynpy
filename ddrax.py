import pandas as pd
import string
#import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumulative_trapezoid
from numba import vectorize, jit
import signal
from nuc import *

#assuming internuclear vectors (dx,dy,dz)
def V0(v):
    return v['dz']

def V1(v):
    return -1/np.sqrt(2)*complex(v['dx'],v['dy'])

def V_1(v):
    return 1/np.sqrt(2)*complex(v['dx'],-v['dy'])

def R20(r):
    return 2/np.sqrt(6)*(V1(r)*V_1(r) + V0(r)**2)

def R21(r):
    return 2/np.sqrt(2)*V0(r)*V1(r)

def R2_1(r):
    return 2/np.sqrt(2)*V0(r)*V_1(r)

def R22(r):
    return V1(r)**2

def R2_2(r):
    return V_1(r)**2

def X(r):
    return(r['dr']*np.sin(r[r'$\theta$'])*np.cos(r['$\phi$']))
def Y(r):
    return(r['dr']*np.sin(r[r'$\theta$'])*np.sin(r['$\phi$']))
def Z(r):
    return(r['dr']*np.cos(r[r'$\theta$']))


def cart_to_spatial(cartdf):
    try:
        time = cartdf['time']
    except KeyError:
        time=None
    return pd.DataFrame.from_dict({"frame":cartdf['frame'], "time":time,"symbol":cartdf['symbol0'],
                                   "label":cartdf['label0'], "$R_{2,-2}$":cartdf.apply(R2_2, axis=1),
                                   "$R_{2,-1}$":cartdf.apply(R2_1, axis=1), "$R_{2,0}$":cartdf.apply(R20, axis=1),
                                   "$R_{2,1}$":cartdf.apply(R21, axis=1), "$R_{2,2}$":cartdf.apply(R22, axis=1)})

def cart_to_spherical(atom_two):
    return pd.DataFrame.from_dict({"frame":atom_two['frame'], "time":atom_two['time'],"symbol0":atom_two['symbol0'],
                                   "symbol1":atom_two['symbol1'],"label0":atom_two['label0'], "label1":atom_two['label1'],
                                   "molecule0":atom_two['molecule0'], "molecule1":atom_two['molecule1'],
                                   r"$\theta$":atom_two.apply(Theta, axis=1),"$\phi$":atom_two.apply(Phi, axis=1),
                                   "dr":atom_two['dr']})

def spherical_to_cart(spherical):
    return pd.DataFrame.from_dict({"frame":spherical['frame'], "time":spherical['time'],"symbol0":spherical['symbol0'],
                                   "symbol1":spherical['symbol1'],"label0":spherical['label0'], "label1":spherical['label1'],
                                   "molecule0":spherical['molecule0'], "molecule1":spherical['molecule1'],
                                   "dx":spherical.apply(X, axis=1),"dy":spherical.apply(Y, axis=1),
                                   "dz":spherical.apply(Z,axis=1)})

#Spherical Harmonics:

@vectorize(nopython=True)
def Theta(dz,dr):
    return np.arccos(dz/dr)
#def Phi(r):
#    try:
#        return np.arctan(r['dy']/r['dx'])
#    except ZeroDivisionError:
#        return 0
@vectorize(nopython=True)
def Phi(dy,dx):
    return np.arctan2(dy,dx)
@vectorize(nopython=True)
def Y20(theta):
    return (1/4)*np.sqrt(5/np.pi)*(3*np.cos(theta)**2 - 1)
@vectorize(nopython=True)
def Y21(theta,phi):
    return (-1/2)*np.sqrt(15/(2*np.pi))*np.sin(theta)*np.cos(theta)*np.exp(complex(0,phi))
@vectorize(nopython=True)
def Y22(theta,phi):
    return (1/4)*np.sqrt(15/(2*np.pi))*np.sin(theta)**2*np.exp(complex(0,2*phi))
@vectorize(nopython=True)
def F0(r,Y):
    return np.sqrt(4*np.pi/5)*Y*r
@vectorize(nopython=True)
def F1(r,Y):
    return np.sqrt(4*np.pi/5)*Y*r

@vectorize(nopython=True)
def F2(r,Y):
    return np.sqrt(4*np.pi/5)*Y*r

#def dipolar_interaction(atom_two_spherical):
#    return pd.DataFrame.from_dict({"frame":atom_two_spherical['frame'], "time":atom_two_spherical['time'],"symbol0":atom_two_spherical['symbol0'],
#                                   "symbol1":atom_two_spherical['symbol1'],"label0":atom_two_spherical['label0'], "label1":atom_two_spherical['label1'],
#                                   "molecule0":atom_two_spherical['molecule0'], "molecule1":atom_two_spherical['molecule1'],
#                                   "$F_{2,0}$":atom_two_spherical.apply(F0, axis=1), "$F_{2,1}$":atom_two_spherical.apply(F1, axis=1),
#                                  "$F_{2,2}$":atom_two_spherical.apply(F2, axis=1)})

def cart_to_dipolar_from_df(atom_two):
    theta = np.vectorize(Theta)(atom_two['dz'],atom_two['dr'])
    phi = np.vectorize(Phi)(atom_two['dy'],atom_two['dx'])
    
    y20 = np.vectorize(Y20)(theta)
    y21 = np.vectorize(Y21)(theta,phi)
    y22 = np.vectorize(Y22)(theta,phi)
    
    r3 = atom_two['dr']**-3
    f0 = np.vectorize(F0)(r3,y20)
    f1 = np.vectorize(F1)(r3,y21)
    f2 = np.vectorize(F2)(r3,y22)

    Fdf = pd.DataFrame.from_dict({"time":atom_two['time'], "label0":atom_two['label0'], "label1":atom_two['label1'], "$r^{-3}$":r3,
        "$Y_{2,0}$":y20,"$Y_{2,1}$":y21,"$Y_{2,2}$":y22,"$F_{2,0}$":f0,"$F_{2,1}$":f1,"$F_{2,2}$":f2})

    return Fdf


def dipolar_var(DD,symbol='H',columns_in=['$F_{2,0}$', '$F_{2,1}$', '$F_{2,2}$'],
              columns_out=['$\langle F^{2}_0\\rangle$', '$\langle F^{2}_1\\rangle$', '$\langle F^{2}_2\\rangle$']):
    s = nuc_df.loc['I',symbol]
    gamma = nuc_df.loc['gamma',symbol]
    C_d = (sp.constants.mu_0/(4*sp.constants.pi))**2*(gamma**4)*(constants.hbar**2)
    s_const = s*(s+1)
    au_m = 5.29177e-11
    F2  = pd.DataFrame((DD[columns_in]**2).mean()).T
    #F2 *= C_d*s_const*au_m**(-6)
    F2.rename(index=str,columns = {columns_in[0]:columns_out[0],columns_in[1]:columns_out[1],columns_in[2]:columns_out[2]},inplace=True)
    return F2

# def dipolar(spec_dens):
#     s = nuc_df.loc['I',spec_dens['symbol']]
#     gamma = nuc_df.loc['gamma',spec_dens['symbol']]
#     C_d = (2/3)*(gamma**4)*(constants.hbar**2)
#     s_const = s*(s+1)
#     au_m = 5.29177e-11
    
#     g0 = 4*spec_dens['$g_{2,2}$'] + spec_dens['$g_{2,1}$'] + spec_dens['$g_{2,-1}$'] + 4*spec_dens['$g_{2,-2}$']
#     g1 = 2*spec_dens['$g_{2,-2}$'] + 3*spec_dens['$g_{2,-1}$'] + 2*spec_dens['$g_{2,1}$'] + 3*spec_dens['$g_{2,0}$']
#     g = 10*spec_dens['$g_{iso}$']
    
#     longitudinal = C_d*s_const*g0*au_m*1e-12
#     transverse = C_d*s_const*g1*au_m*1e-12
#     isotropic = C_d*s_const*g*au_m*1e-12
    
#     #print(s)
#     #print(spec_dens['symbol'])
    
#     print(C_d)
#     print(s_const)
#     print(g0,g1,g)
#     #print(longitudinal,transverse,isotropic)

#     return pd.DataFrame({r'$\frac{1}{T_{1}}$':longitudinal, r'$\frac{1}{T_{2}}$':transverse, r'$\frac{1}{T_{iso}}$':isotropic}, index=[spec_dens['symbol']])


def fourier_T(f):
    n = len(f)
    F = np.fft.fft(f)[range(n//2)]
    #print(df.time.diff().values[1])
    return F
def fourier_freq(t):
    n = len(t)
    dt = t.diff().values[1]
    freq = np.fft.fftfreq(n,dt)[range(n//2)]
    return freq

def spec_dens(acf):
    freq = fourier_freq(acf['time'])
    J0 = fourier_T(acf['$G_{2,0}$'])
    J1 = fourier_T(acf['$G_{2,1}$'])
    J2 = fourier_T(acf['$G_{2,2}$'])
    Jdf = pd.DataFrame({'$\omega$':freq, '$log(\omega)$':np.log10(freq), '$\omega^{\\frac{1}{2}}$':np.sqrt(freq), '$J_{0}(\omega)$':J0, '$J_{1}(\omega)$':J1, '$J_{2}(\omega)$':J2})
    return Jdf

def spec_dens_extr_narr(acf,dt=None,columns_in=['$G_{2,0}$', '$G_{2,1}$', '$G_{2,2}$'],columns_out=['$j_{2,0}$', '$j_{2,1}$', '$j_{2,2}$'], cutoff=False, cutoff_tol=1e-3):
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
               
        g = 2*acf.iloc[:bounds[l]][columns_in].apply(sp.integrate.simps,x=acf.iloc[:bounds[l]]['time'])
        #g = pd.DataFrame([np.trapz(acf.iloc[:bounds[l]][m], x=acf.iloc[:bounds[l]]['time']) for l,m in enumerate(columns_in)]).transpose()
        #g.columns = columns_out
        #g['$g_{iso}$'] = np.mean([g[m] for m in columns_out])
        #g['symbol'] = acf['symbol'].iloc[0]
        
    else:
        if dt:
            g = 2*acf[columns_in].apply(sp.integrate.simps, dx=dt)
        else:
            g = 2*acf[columns_in].apply(sp.integrate.simps, x=acf['time'])
        
    g.index = columns_out
    g['$\langle F^{2}_{0}\\rangle$'] = acf.iloc[0]['$G_{2,0}$']
    g['$\langle F^{2}_{1}\\rangle$'] = acf.iloc[0]['$G_{2,1}$']
    g['$\langle F^{2}_{2}\\rangle$'] = acf.iloc[0]['$G_{2,2}$']
    g['$\langle F(0)^{2}\\rangle$'] = (g['$\langle F^{2}_{1}\\rangle$']+g['$\langle F^{2}_{2}\\rangle$'])/2
    g['$\\tau_{0}$'] = g['$j_{2,0}$']/g['$\langle F^{2}_{0}\\rangle$']
    g['$\\tau_{1}$'] = g['$j_{2,1}$']/g['$\langle F^{2}_{1}\\rangle$']
    g['$\\tau_{2}$'] = g['$j_{2,2}$']/g['$\langle F^{2}_{2}\\rangle$']
    g['$\\tau_{c}$'] =(g['$\\tau_{1}$']+g['$\\tau_{2}$'])/2
    
    return g

def dipolar(spec_dens,symbol1,symbol2='H',extr_narr=True,single_J=False,larmor_freq=25):
    nuc_df = pd.DataFrame.from_dict(nuc)
    s = nuc_df.loc['I',symbol1]
    gamma1 = nuc_df.loc['gamma',symbol1]
    gamma2 = nuc_df.loc['gamma',symbol2]
    C_d = (sp.constants.mu_0/(4*sp.constants.pi))**2*(gamma1**2)*(gamma2**2)*(constants.hbar**2)
    s_const = s*(s+1)
    au_m = 5.29177e-11
    
    omega_0 = larmor_freq * 1e6
    
    if single_J:
        j1 = j2 = spec_dens
        R = np.real(C_d*s_const*(j1+(4*j2))*au_m**(-6)*1e-12)
        return pd.DataFrame({'$\\frac{1}{T_{1}}$':R}, index=[symbol])
    
    elif extr_narr:
        #j0 = spec_dens['$j_{2,0}$']
        #j1 = spec_dens['$j_{2,1}$']
        #j2 = spec_dens['$j_{2,2}$']
        j0 = j1 = j2 = spec_dens[['$j_{2,0}$','$j_{2,1}$','$j_{2,2}$']].mean()
        tc = spec_dens['$\\tau_{c}$']
        f = C_d*s_const*au_m**(-6)*spec_dens['$\langle F^{2}_{0}\\rangle$']
        zf = None
    else:
        m0 = (np.real(J.iloc[2]['$J_{0}(\omega)$']) - np.real(J.iloc[1]['$J_{0}(\omega)$']))/(np.real(J.iloc[2]['$\omega^{\\frac{1}{2}}$']) - np.real(J.iloc[1]['$\omega^{\\frac{1}{2}}$']))
        b0 = np.real(J.iloc[2]['$J_{0}(\omega)$']) - m0*np.real(J.iloc[2]['$\omega^{\\frac{1}{2}}$'])
        j0 = b0 - m0*np.sqrt(omega_0)
        j02 = b0 - m0*np.sqrt(2*omega_0)
        
        zf = C_d*s_const*(j0+(4*j02))*au_m**(-6)*1e-12
        
        m1 = (np.real(J.iloc[2]['$J_{1}(\omega)$']) - np.real(J.iloc[1]['$J_{1}(\omega)$']))/(np.real(J.iloc[2]['$\omega^{\\frac{1}{2}}$']) - np.real(J.iloc[1]['$\omega^{\\frac{1}{2}}$']))
        b1 = np.real(J.iloc[2]['$J_{1}(\omega)$']) - m1*np.real(J.iloc[2]['$\omega^{\\frac{1}{2}}$'])
        j1 = b1 - m1*np.sqrt(omega_0)
        
        m2 = (np.real(J.iloc[2]['$J_{2}(\omega)$']) - np.real(J.iloc[1]['$J_{2}(\omega)$']))/(np.real(J.iloc[2]['$\omega^{\\frac{1}{2}}$']) - np.real(J.iloc[1]['$\omega^{\\frac{1}{2}}$']))
        b2 = np.real(J.iloc[2]['$J_{2}(\omega)$']) - m2*np.real(J.iloc[2]['$\omega^{\\frac{1}{2}}$'])
        j2 = b2 - m2*np.sqrt(2*omega_0)
        
        #j0 = spec_dens.iloc[0]['$J_{0}(\omega)$']
        #j1 = spec_dens.iloc[0]['$J_{1}(\omega)$']
        #j2 = spec_dens.iloc[0]['$J_{2}(\omega)$']

    #print(j0,j1,j2)
    if symbol1 == symbol2:
        R = np.real(C_d*s_const*(j1+(4*j2))*au_m**(-6)*1e-12)
    else:
        R = (2/3)*C_d*s_const*10*np.real(j0)*au_m**(-6)*1e-12

    return R, tc, f

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

def correlate(df,columns_in=['$Y_{2,0}$','$Y_{2,1}$','$Y_{2,2}$','$F_{2,0}$', '$F_{2,1}$', '$F_{2,2}$'],
              columns_out=['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$','$G_{2,0}$', '$G_{2,1}$', '$G_{2,2}$'], 
              pass_columns=['time','label0','label1']):
    acf = df[columns_in].apply(wiener_khinchin)
    acf.columns = columns_out
    r_m = df['$r^{-3}$'] - df['$r^{-3}$'].mean()
    acf['$G_{r}$'] = wiener_khinchin(r_m)
    acf[pass_columns] = df[pass_columns]
    return acf

def llse(pdf):
    f = np.array(pdf)
    #print(f)
    n = len(f)
    err = np.empty((n, ), dtype=np.float64)
    for i in range(n):
        err[i] = 1
        for k in range(i):
            err[i] += 2*f[k]**2
    err /= n
    err = err**(1/2)
    return err

def fse(acf,columns_in=['$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$'], columns_out=['$e_{2,-2}$', '$e_{2,-1}$', '$e_{2,0}$', '$e_{2,1}$', '$e_{2,2}$']):
    #print(acf)
    err = acf[columns_in].apply(llse)
    err.columns = columns_out
    err[['frame','time','label', 'symbol']] = acf[['frame','time','label', 'symbol']]
    return err

def trapzvar(var,width):
    #res = width * np.sum(var) - (var[0]+var[-1])
    #return res
    return width/2*(2*np.sum(var[1:-1]) + var[0] + var[-1])

def normal_factor(acf,columns_in=['$f_{2,-2}$', '$f_{2,-1}$', '$f_{2,0}$', '$f_{2,1}$', '$f_{2,2}$'],columns_out=['$\sigma_{2,-2}$', '$\sigma_{2,-1}$', '$\sigma_{2,0}$', '$\sigma_{2,1}$', '$\sigma_{2,2}$']):
    V = acf[columns_in].iloc[0]
    V.index = columns_out
    V[r'$\langle V(0)^2\rangle$'] = V.sum()
    V['symbol'] = acf['symbol'].iloc[0]
    
    return V

def correlation_time(spec_dens,norm):
    tau = pd.DataFrame.from_dict({r'$\tau_{2,'+str(m)+'}$':spec_dens['$g_{2,'+str(m)+'}$']/norm['$\sigma_{2,'+str(m)+'}$'] for m in range(-2,3)})
    tau[r'$\tau_{c}$'] = spec_dens['$g_{iso}$']*5/norm[r'$\langle V(0)^2\rangle$']
    tau['symbol'] = spec_dens['symbol']
    return tau

def compute_nu(nuc):
    I_k = nuc_df[nuc['symbol']]['I']
    eta_k = nuc['eta']
    eq_k = constants.e*nuc_df[nuc['symbol']]['Q']
    v33_k = nuc['V33']
    nu = 3/40*(2*I_k+3)/(I_k**2*(2*I_k-1))*(1+eta_k**2/3)*(eq_k/constants.hbar*v33_k)**2
    
    return nu

def cart_rax(nuc):
    I_k = nuc_df[nuc['symbol']]['I']
    eq_k = constants.e*nuc_df[nuc['symbol']]['Q']
    g = nuc['g']
    au_q = 9.71736408e21
    r = 3/40*(2*I_k+3)/(I_k**2*(2*I_k-1))*(eq_k/constants.hbar)**2*g*au_q**2*1e-12
    return r

def reorder_prnc_comp(nuc,columns=['V11','V22','V33']):
    ordered = sorted(nuc[columns],key=lambda x: abs(float(x)))
    onuc = nuc.copy()
    onuc[columns] = ordered
    return onuc

@jit(nopython=True, nogil=True)
def things(r):
    n = len(r)
    res = np.empty((n, ), dtype=np.float64)
    err = res.copy()
    conj = np.conjugate(r)
    for i in range(n):
        summation = 0
        m = n - i
        mr = np.sqrt(m)
        relsemj = (np.std(conj[:m])/mr)/np.mean(conj[:m])
        relsemij = (np.std(r[i:n])/mr)/np.mean(r[i:n])
        error = 0
        for j in range(n-i):
            term = conj[j]*r[i + j]
            error += term**2
            summation += term
        res[i] = summation
        err[i] = np.sqrt(error*(relsemj**2 + relsemij**2))
    return res, err

def llse(acf):
    n = len(acf)
    err = np.empty((n, ), dtype=np.float64)
    for i in range(n):
        err[i] = 1
        for k in range(i):
            err[i] += 2*acf[k]**2
    err /= n
    err = err**(1/2)
    return err

##Harmon Muller##

mdict = {'N':14.01, 'C':12.01, 'H':1.01, 'O':15.999}
def CoM(df):
    sumx = np.sum([mdict[sym]*x for (sym,x) in zip(df.symbol,df.x)])
    sumy = np.sum([mdict[sym]*y for (sym,y) in zip(df.symbol,df.y)])
    sumz = np.sum([mdict[sym]*z for (sym,z) in zip(df.symbol,df.z)])
    sumc = np.array([sumx,sumy,sumz])

    summ = np.sum([mdict[sym] for sym in df.symbol])

    com = sumc/summ

    COM = pd.DataFrame({'x':com[0],'y':com[1],'z':com[2],'frame':df.reset_index().frame[0],'symbol':'O'}, index=[0])
    return COM

#Sholl: ALl input distance units should be Angstrom. Returns spectral dens in ps*au^-6
def HM(n,sigma,D,b,g,sym='H',returnI=False):
    g.loc[:,'g1'] = g.loc[:,'$g(r)$']*(sigma/g.index)**2
    g.loc[:,'g2'] = g.loc[:,'$g(r)$']*(sigma/g.index)**4

    I1 = (1/sigma)*sp.integrate.simps(g.g1,x=g.index)
    I2 = (1/sigma)*sp.integrate.simps(g.g2,x=g.index)

    II = 5*I1 - 3*I2

    J = 1/20*(n*8*np.pi/15/(sigma*D))*((sigma/b)*(5-sigma**2/b**2) + II)
    J = J*(0.529177)**6

    if returnI:
        return J,II
    else:
        return J
