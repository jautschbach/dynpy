import signal as sig
#from dynpy import worker_init
import pandas as pd
import string
#import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumtrapz
from numba import vectorize, guvectorize, float64, complex128
import os
import sys
#import exa
#import exatomic
#exa.logging.disable(level=10)
#exa.logging.disable(level=20)
#from exatomic import qe
import math
from multiprocessing import Pool, cpu_count
import psutil

def worker_init(parent_id):
    #print(parent_id)
    #parent = psutil.Process(parent_id)
    #print(parent)
    #print(parent.children())
    def sig_int(signal_num, frame):
        print(parent_id)
        print('signal: %s' % signal_num)
        parent = psutil.Process(parent_id)
        for child in parent.children():
            print(child.pid)
            if child.pid != os.getpid():
                print("killing child: %s" % child.pid)
                child.kill()
        print("killing parent: %s" % parent_id)
        parent.kill()
        print("suicide: %s" % os.getpid())
        psutil.Process(os.getpid()).kill()
        os._exit(1)
    sig.signal(sig.SIGINT, sig_int)

def applyParallel(func,dfGrouped,cpus=cpu_count(),**kwargs):
    values = kwargs.values()
    with Pool(cpus) as p:
        ret_list = p.starmap(func, [(group,*values) for name, group in dfGrouped])
    return pd.concat(ret_list)
def applyParallel2(func,dfGrouped1,dfGrouped2,cpus=cpu_count(),*args):
    #print([(group1,group2,*args) for group1,group2 in zip(dfGrouped1,dfGrouped2)])
    with Pool(cpus) as p:
        ret_list = p.starmap(func, [(group1[1],group2[1],*args) for group1,group2 in zip(dfGrouped1,dfGrouped2)])
    return pd.concat(ret_list)

def applyParallel3(func,dfGrouped1,dfGrouped2,dfGrouped3,cpus=cpu_count(),**kwargs):
    parent_id = os.getppid()
    #sig.signal(sig.SIGINT, signal_handler)
    values = kwargs.values()
    with Pool(cpus,worker_init(parent_id)) as p:
        try:
            ret_list = p.starmap_async(func, [(group1[1],group2[1],group3[1],*values) for group1,group2,group3 in zip(dfGrouped1,dfGrouped2,dfGrouped3)]).get()
        except KeyboardInterrupt:
        #    p.terminate()
            sys.exit(1)
    #return ret_list
    ret = list(zip(*ret_list))
    #print(ret)
    return pd.concat(ret[0]),pd.concat(ret[1]),pd.concat(ret[2])
    #return pd.concat(ret_list)

def applyParallel_D(func,dfGrouped1,dfGrouped2,dfGrouped3,cpus=cpu_count(),**kwargs):
    values = kwargs.values()
    #print([(group1,group2,*args) for group1,group2 in zip(dfGrouped1,dfGrouped2)])
    with Pool(cpus) as p:
        ret_list = p.starmap_async(func, [(group1[1],group2[1],group3[1],*values) for group1,group2,group3 in zip(dfGrouped1,dfGrouped2,dfGrouped3)]).get()
    #return ret_list
    ret = list(zip(*ret_list))
    #print(ret)
    if kwargs['return_euler']:
        return pd.concat(ret[0]),pd.concat(ret[1]),pd.concat(ret[2]),pd.concat(ret[3])
    else:
        return pd.concat(ret[0]),pd.concat(ret[1]),pd.concat(ret[2])
    #return pd.concat(ret_list)

def Theta(r):
    return np.arccos(r['dz'].values[0]/r['dr'].values[0])
def Phi(r):
    return np.arctan2(r['dy'].values[0],r['dx'].values[0])

def cart_to_spherical(atom_two):
    return pd.DataFrame.from_dict({"frame":atom_two['frame'], "time":atom_two['time'],
                                   "molecule":atom_two['molecule'],r"$\theta$":atom_two.apply(Theta, axis=1),
                                   r"$\phi$":atom_two.apply(Phi, axis=1)})

def Y10(theta):
    return (1/2)*np.sqrt(3/np.pi)*(np.cos(theta))

def Y20(theta):
    return (1/4)*np.sqrt(5/np.pi)*(3*np.cos(theta)**2 - 1)

def spherical_harmonics(spherical):
    return pd.DataFrame.from_dict({"frame":spherical['frame'], "time":spherical['time'],
                               "molecule":spherical['molecule'],r"$Y_{1,0}$":Y10(spherical[r"$\theta$"]),
                               r"$Y_{2,0}$":Y20(spherical[r"$\theta$"])})


#@jit(nopython=True,parallel=True)
def plane_norm(a,b):
   #a = np.array(df.iloc[0][['dx','dy','dz']].astype(float))
   #b = np.array(df.iloc[1][['dx','dy','dz']].astype(float))
   c = np.cross(a,b)
   cc = np.linalg.norm(c)
   norm = c/cc
   return norm

#@guvectorize(['void(float64[:],float64[:],float64)'],'(n),(n)->()',nopython=True)
#@jit(nopython=True,parallel=True)
def vec_angle(a,b):
   #det = np.dot(n,np.cross(a,b))
   dot = np.dot(a,b)
   aa = np.linalg.norm(a)
   bb = np.linalg.norm(b)
   c = aa*bb
   d = dot/c
   inner = np.arccos(d)
   return inner

def rel_C(df):
    rel = df.copy()
    rel.loc[:,['x','y','z']] = rel.loc[:,['x','y','z']].to_numpy() - rel[rel['symbol']=='C'][['x','y','z']].to_numpy()
    return rel

@vectorize([float64(float64, float64)])
def AA(y,z):
    return y**2 + z**2
@vectorize([float64(float64, float64)])
def AB(x,y):
    return -x*y
@vectorize([float64(float64, float64)])
def AC(x,z):
    return -x*z
@vectorize([float64(float64, float64)])
def BB(x,z):
    return x**2 + z**2
@vectorize([float64(float64, float64)])
def BC(y,z):
    return -y*z
@vectorize([float64(float64, float64)])
def CC(x,y):
    return x**2 + y**2

def make_R(df):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    aa = np.sum(AA(y,z))
    ab = np.sum(AB(x,y))
    ac = np.sum(AC(x,z))
    bb = np.sum(BB(x,z))
    bc = np.sum(BC(y,z))
    cc = np.sum(CC(x,y))
    return np.array([[aa,ab,ac],[ab,bb,bc],[ac,bc,cc]])

@vectorize([float64(float64, float64, float64)])
def Ixx(y,z,m):
    return m*(y**2 + z**2)
@vectorize([float64(float64, float64, float64)])
def Ixy(x,y,m):
    return -m*x*y
@vectorize([float64(float64, float64, float64)])
def Ixz(x,z,m):
    return -m*x*z
@vectorize([float64(float64, float64, float64)])
def Iyy(x,z,m):
    return m*(x**2 + z**2)
@vectorize([float64(float64, float64, float64)])
def Iyz(y,z,m):
    return -m*y*z
@vectorize([float64(float64, float64, float64)])
def Izz(x,y,m):
    return m*(x**2 + y**2)

def make_I(df):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    z = df['z'].to_numpy()
    m = df['mass'].to_numpy()
    aa = np.sum(Ixx(y,z,m))
    ab = np.sum(Ixy(x,y,m))
    ac = np.sum(Ixz(x,z,m))
    bb = np.sum(Iyy(x,z,m))
    bc = np.sum(Iyz(y,z,m))
    cc = np.sum(Izz(x,y,m))
    return np.array([[aa,ab,ac],[ab,bb,bc],[ac,bc,cc]])

def cross(pos,vel):
    #print(np.array(pos[['x','y','z']]))
    #print(np.array(vel[['x','y','z']]))
    #print(pos.values)
    res = np.sum(np.cross(np.array(pos[['x','y','z']]),np.array(vel[['x','y','z']])), axis=0)
    return res

def determine_center(attr, coords):
    """Determine the center of the molecule with respect to
    the given attribute data. Used for the center of nuclear
    charge and center of mass."""
    center = 1/np.sum(attr)*np.sum(np.multiply(np.transpose(coords), attr), axis=1)
    center = pd.Series(center, index=['x', 'y', 'z'])
    return center
def rel_center(coord):
    return coord[['x','y','z']] - determine_center(coord.mass,coord[['x','y','z']])

def wiener_khinchin(f):
   #Wiener-Khinchin theorem
   real = pd.Series(np.real(f)).interpolate()
   imag = pd.Series(np.imag(f)).interpolate()
   f = pd.Series([complex(r,i) for r,i in zip(real,imag)])
   N = len(f)
   fvi = np.fft.fft(f,n=2*N)
   acf = np.real(np.fft.ifft(fvi * np.conjugate(fvi))[:N])
   acf = acf/N
   return acf
def correlate(df,columns_in=[r'$\omega$',r'$\omega_x$',r'$\omega_y$',r'$\omega_z$'],columns_out=['$G$','$G_x$','$G_y$','$G_z$'], pass_columns=['time','molecule0']):
    acf = df[columns_in].apply(wiener_khinchin)
    acf.columns = columns_out
    acf[pass_columns] = df[pass_columns]
    return acf

#@jit(nopython=True,parallel=True)
def Proj(a,b):
    bb = np.linalg.norm(b)
    bbb = b/bb
    c = np.dot(a,bbb)
    proj = a - c/bb**2*b
    return proj

def vec_angle2(a,b,n):
    N = np.cross(a,b)
    #print(N)
    #n = n/la.norm(n)
    det = np.dot(n,N)
    #print(det)
    dot = np.dot(a,b)
    angle = np.arctan2(det,dot)
    if det >=0:
        return angle
    elif det < 0:
        return 2*np.pi+angle
def vec_angle_b(a,b,n):
    N = np.cross(a,b)
    #print(N)
    n = n/la.norm(n)
    det = np.dot(n,N)
    #print(det)
    dot = np.dot(a,b)
    angle = np.arctan2(det,dot)
    return angle

def node(mol_ax,rel_z=np.array([0,0,1])):
    N = np.cross(rel_z,mol_ax[['zx','zy','zz']].values[0])
    return N

#def alpha(N,rel_y=np.array([0,1,0]),rel_z=np.array([0,0,1])):
#    return vec_angle2(rel_y,N,rel_z)
#def beta(mol_ax,N,rel_z=np.array([0,0,1])):
#    return vec_angle2(rel_z,np.array(mol_ax[['zx','zy','zz']].values[0]),N)
#def gamma(mol_ax,N):
#    return vec_angle2(N,np.array(mol_ax[['yx','yy','yz']].values[0]),np.array(mol_ax[['zx','zy','zz']].values[0]))

def alpha(R):
    return np.arctan2(R[1][2],R[0][2])
def beta(R):
    return np.arctan2(np.sqrt(1-R[2][2]**2),R[2][2])
def gamma(R):
    return np.arctan2(R[2][1],-R[2][0])

@vectorize([float64(float64)])
def D00(beta):
    return np.cos(beta)
@vectorize([complex128(float64, float64)])
def D0_1(beta,gamma):
    return -np.sqrt(1/2)*np.sin(beta)*np.exp(complex(0,gamma))
@vectorize([complex128(float64, float64)])
def D01(beta,gamma):
    return np.sqrt(1/2)*np.sin(beta)*np.exp(complex(0,-gamma))
@vectorize([complex128(float64, float64)])
def D10(alpha,beta):
    return -np.exp(complex(0,-alpha))*np.sqrt(1/2)*np.sin(beta)
@vectorize([complex128(float64, float64)])
def D_10(alpha,beta):
    return np.exp(complex(0,alpha))*np.sqrt(1/2)*np.sin(beta)
@vectorize([complex128(float64, float64, float64)])
def D11(alpha,beta,gamma):
    return np.exp(complex(0,-alpha))*(1/2)*(1+np.cos(beta))*np.exp(complex(0,-gamma))
@vectorize([complex128(float64, float64, float64)])
def D_1_1(alpha,beta,gamma):
    return np.exp(complex(0,alpha))*(1/2)*(1+np.cos(beta))*np.exp(complex(0,gamma))
@vectorize([complex128(float64, float64, float64)])
def D1_1(alpha,beta,gamma):
    return np.exp(complex(0,-alpha))*(1/2)*(1-np.cos(beta))*np.exp(complex(0,gamma))
@vectorize([complex128(float64, float64, float64)])
def D_11(alpha,beta,gamma):
    return np.exp(complex(0,alpha))*(1/2)*(1-np.cos(beta))*np.exp(complex(0,-gamma))

#def Euler(mol_ax,pass_columns=None):
#    N = node(mol_ax)
#    a = alpha(N)
#    b = beta(mol_ax,N)
#    g = gamma(mol_ax,N)
#    Euler_df = pd.DataFrame({"$\\alpha$":a,"$\\beta$":b,"$\gamma$":g},index=mol_ax.index)
#    Euler_df[pass_columns] = mol_ax[pass_columns]
#    return Euler_df
def Wigner(mol_ax,mol_ax_init,mol_label,frame,return_euler=False):
    #print("mol_label= "+str(mol_label))
    init_ax = mol_ax_init[mol_label]
    R = la.solve(init_ax,mol_ax)
    #N = node(mol_ax,rel_z=init_mol_z)
    #a = alpha(N,rel_y=init_mol_y,rel_z=init_mol_z)
    a = alpha(R)
    #b = beta(mol_ax,N,rel_z=init_mol_z)
    b = beta(R)
    #g = gamma(mol_ax,N)
    g = gamma(R)
    #print(a,b,g)
    Wigner = pd.DataFrame({"frame":frame,"molecule_label":mol_label,"$D_{-1,-1}$":D_1_1(a,b,g), "$D_{-1,0}$":D_10(a,b), "$D_{-1,1}$":D_11(a,b,g), "$D_{0,-1}$":D0_1(b,g),"$D_{0,0}$":D00(b), "$D_{0,1}$":D01(b,g), "$D_{1,-1}$":D1_1(a,b,g), "$D_{1,0}$":D10(a,b), "$D_{1,1}$":D11(a,b,g)},index=[0])
    #if pass_columns:
    #    Wigner[pass_columns] = mol_ax[pass_columns]
    if return_euler:
        Euler_df = pd.DataFrame({"frame":frame,"molecule_label":mol_label,r"$\\alpha$":a,r"$\\beta$":b,r"$\gamma$":g},index=[0])
        #if pass_columns:
        #    Euler_df[pass_columns] = mol_ax[pass_columns]
        return Wigner, Euler_df
    else:
        return Wigner

@vectorize([complex128(float64, float64)])
def J_1(x,y):
    return 1/np.sqrt(2)*complex(x,-y)
@vectorize([complex128(float64, float64)])
def J1(x,y):
    return -1/np.sqrt(2)*complex(x,y)

def cart_to_spatial(cartdf,pass_columns=['frame','molecule','molecule_label']):
    #try:
    #    time = cartdf['time']
    #except KeyError:
    #    time=None
    spatial = pd.DataFrame({"$J_{0}$":cartdf['z'].values, "$J_{-1}$":J_1(cartdf['x'].values,cartdf['y'].values),
                                   "$J_{1}$":J1(cartdf['x'].values,cartdf['y'].values)})
    if pass_columns:
        spatial[pass_columns] = cartdf[pass_columns]

    return spatial

def K11(J,D,c_a,c_d,pass_columns=['frame','molecule','molecule_label']):
    j = (-1)*c_a*J['$J_{-1}$']
    cg={-1:np.sqrt(1/6),0:np.sqrt(1/2),1:1}
    dj=0
    for mu in range(-1,2):
        #print('$D_{0,'+str(-(1+mu))+'}$')
        dj += (-1)**(mu)*cg[mu]*D['$D_{0,'+str(-(1+mu))+'}$']*J['$J_{'+str(mu)+'}$']
    DJ = np.sqrt(2/3)*c_d*dj
    K = j + DJ
    Kdf = pd.DataFrame({'$K_{1,1}$':K})
    Kdf[pass_columns]=J[pass_columns]
    return Kdf

def mol_fixed_coord(mol,mol_type,**kwargs):
    for key, value in kwargs.items():
        if key=='methyl_indeces':
            methyl_indeces = value
    
    if mol_type.casefold()=="acetonitrile":
        CN = mol[(mol['mol-atom_index0'] == 0) & (mol['mol-atom_index1'] == 64)][['dx','dy','dz']].values.astype(float)[0]
        z = CN/la.norm(CN)
        CH = mol[(mol['mol-atom_index0']==0) & (mol['mol-atom_index1']==96)][['dx','dy','dz']].values.astype(float)[0]
        x = plane_norm(z,CH)
        #x = mol.iloc[2][['dx','dy','dz']].values.astype(float)
        y = plane_norm(z,x)
        #print(x,y,z)
        return np.array([x,y,z]).T
    
    elif mol_type.casefold()=="methane":
        z = mol.iloc[0][['dx','dy','dz']].values.astype(float)/la.norm(mol.iloc[0][['dx','dy','dz']].values.astype(float))
        x = plane_norm(z,mol.iloc[1][['dx','dy','dz']].values.astype(float))
        #x = mol.iloc[2][['dx','dy','dz']].values.astype(float)
        y = plane_norm(z,x)
        #print(np.array(x),y,z)
        return np.array([x,y,z]).T
    
    elif mol_type.casefold()=="methyl":
        R = methyl_indeces[0]
        C = methyl_indeces[1]
        H1 = methyl_indeces[2]
        H2 = methyl_indeces[3]
        H3 = methyl_indeces[4]

        RC = mol[(mol['mol-atom_index0']==R) & (mol['mol-atom_index1']==C)][['dx','dy','dz']].values.astype(float)[0]
        CH1 = mol[(mol['mol-atom_index0']==C) & (mol['mol-atom_index1']==H1)][['dx','dy','dz']].values.astype(float)[0]
        z = RC/la.norm(RC)
        x = plane_norm(z,CH1)
        #x = mol.iloc[2][['dx','dy','dz']].values.astype(float)
        y = plane_norm(z,x)
        #print(np.array(x),y,z)
        return np.array([x,y,z]).T
    
    elif mol_type.casefold()=="water":
        OH1 = mol[(mol['mol-atom_index0']==0) & (mol['mol-atom_index1']==1)][['dx','dy','dz']].values.astype(float)[0]
        OH2 = mol[(mol['mol-atom_index0']==0) & (mol['mol-atom_index1']==2)][['dx','dy','dz']].values.astype(float)[0]
        Z = -(OH1+OH2)/la.norm(OH1+OH2)
        X = plane_norm(Z,OH1)
        Y = plane_norm(Z,X)
        #print(np.array(x),y,z)
        return np.array([X,Y,Z]).T
    else:
        sys.exit("Only molecule types methane, water, and acetonitrile are supported")

def SR_func1(pos,vel,two,mol_type,methyl_indeces=None,rot_mat=np.diag([1,1,1])):
    #print(mol_type)
    #print(methyl_indeces)
    pos[['x','y','z']] = rel_center(pos)
    #print("pos rel to center of mass--- %s seconds ---" % (time.time() - start_time))
    vel[['x','y','z']] = rel_center(vel)
    #print("vel rel to center of mass--- %s seconds ---" % (time.time() - start_time))

    R = make_R(pos)
    #print("construct R mat--- %s seconds ---" % (time.time() - start_time))
    I = make_I(pos)
    #print("construct I mat--- %s seconds ---" % (time.time() - start_time))
    rv = cross(pos,vel)
    #print("cross product--- %s seconds ---" % (time.time() - start_time))
    #print(R,rv)

    o = la.solve(R,rv)
    #print("la.solve--- %s seconds ---" % (time.time() - start_time))
    #o_cart_df = pd.DataFrame(o.reshape((1,3)),columns=['x','y','z'])
    #o_cart_df = o_cart_df.assign(frame=pos.frame.iloc[0],molecule=pos.molecule.iloc[0],molecule_label=pos.molecule_label.iloc[0])
    mol_ax = mol_fixed_coord(two,mol_type,methyl_indeces=methyl_indeces)
    #print("construct mol_fixed--- %s seconds ---" % (time.time() - start_time))
    mol_ax_df = pd.DataFrame(mol_ax.reshape((1,9)),columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz'])
    mol_ax_df = mol_ax_df.assign(frame=pos.frame.iloc[0],molecule=pos.molecule.iloc[0],molecule_label=pos.molecule_label.iloc[0])
    #print("construct mol_ax df--- %s seconds ---" % (time.time() - start_time))
    ax = np.matmul(mol_ax,rot_mat)
    #print("apply rotation if requested--- %s seconds ---" % (time.time() - start_time))
    
    oI = np.matmul(la.inv(ax),o)
    #print("ang vel in mol ax--- %s seconds ---" % (time.time() - start_time))
    o_ax_df = pd.DataFrame(oI.reshape((1,3)),columns=['x','y','z'])
    o_ax_df = o_ax_df.assign(frame=pos.frame.iloc[0],molecule=pos.molecule.iloc[0],molecule_label=pos.molecule_label.iloc[0])
    #print("construct ang vel df--- %s seconds ---" % (time.time() - start_time))
    I_ax = np.matmul(np.matmul(la.inv(ax),I),ax)
    #print("I mat in mol ax--- %s seconds ---" % (time.time() - start_time))
    J_cart = np.matmul(I_ax,oI)
    #print("ang mom in mol ax--- %s seconds ---" % (time.time() - start_time))
    #ax_df = pd.DataFrame(ax.flatten().reshape((1,9)),columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz'])
    #ax_df[['frame','molecule','molecule_label']]=pos[['frame','molecule','molecule_label']]
    J_cart_df = pd.DataFrame(J_cart.reshape((1,3)),columns=['x','y','z'])
    J_cart_df = J_cart_df.assign(frame=pos.frame.iloc[0],molecule=pos.molecule.iloc[0],molecule_label=pos.molecule_label.iloc[0])
    #print("construct ang mom df--- %s seconds ---" % (time.time() - start_time))
    #print(J_cart_df.values)
    #print(two.frame.iloc[0])
    #if two.frame.iloc[0]==30100:
    #    with open("/projects/academic/jochena/adamphil/projects/SR/acetonitrile/check.txt", 'w') as f:
    #        f.write(str(o)+'\n'+str(oI)+'\n'+str(mol_ax)+'\n'+str(ax)+'\n'+str(I)+'\n'+str(I_ax)+'\n'+str(J_cart)+'\n'+str(J_cart_df))
    return mol_ax_df,o_ax_df,J_cart_df

def SR_func2(pos,vel,two,mol_type,mol_ax_init,return_euler=False,rot_mat=np.diag([1,1,1])):
    #sig.signal(sig.SIGINT, signal_handler)
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #print(pos)
    pos[['x','y','z']] = rel_center(pos)
    vel[['x','y','z']] = rel_center(vel)
    frame = pos.frame.iloc[0]
    mol = pos.molecule.iloc[0]
    mol_label = pos.molecule_label.iloc[0]
    R = make_R(pos)
    I = make_I(pos)
    rv = cross(pos,vel)
    #print(R,rv)

    o = la.solve(R,rv)
    
    mol_ax = mol_fixed_coord(two,mol_type)
    ax = np.matmul(mol_ax,rot_mat)
    
    oI = np.matmul(la.inv(ax),o)
    o_ax_df = pd.DataFrame(oI.reshape((1,3)),columns=['x','y','z'])
    o_ax_df = o_ax_df.assign(frame=frame,molecule=mol,molecule_label=mol_label)
    I_ax = np.matmul(np.matmul(la.inv(ax),I),ax)
    J_cart = np.matmul(I_ax,oI)
    J_cart_df = pd.DataFrame(J_cart.reshape((1,3)),columns=['x','y','z'])
    J_cart_df = J_cart_df.assign(frame=frame,molecule=mol,molecule_label=mol_label)
    
    ax_df = pd.DataFrame(ax.flatten().reshape((1,9)),columns=['xx','xy','xz','yx','yy','yz','zx','zy','zz'])
    ax_df = ax_df.assign(frame=frame,molecule=mol,molecule_label=mol_label)
    J_sph_df = cart_to_spatial(J_cart_df,pass_columns=['frame','molecule_label'])
    D_mat,Euler = Wigner(ax,mol_ax_init,mol_label,frame,return_euler=return_euler)

    return J_sph_df,D_mat,Euler,ax_df

def major_ax(df):
    CH = np.array(df.iloc[1].values[:3].astype(float))
    CC = np.array(df.iloc[0].values[:3].astype(float))
    norm = plane_norm(CC,np.array([0,0,1.0]))
    CH_proj = Proj(CH,CC)
    b = vec_angle(CH_proj,norm)
    #frame = df.iloc[0].frame
    time = df.iloc[0].time
    #molecule = df.iloc[0].molecule0
    #print(frame,time,molecule,e[0],e[1],e[2])
    return pd.Series({'time':time,'theta':b})

def minor_ax(df):
    CN = np.array(df.iloc[4].values[:3].astype(float))
    b = vec_angle(CN,np.array([0,0,1.0]))
    c = vec_angle(CN,np.array([0,1.0,0]))
    #frame = df.iloc[0].frame
    time = df.iloc[0].time
    #molecule = df.iloc[0].molecule0
    #print(frame,time,molecule,e[0],e[1],e[2])
    return pd.Series({'time':time,'theta':b,'phi':c})

def adiff(df):
    dtheta = df['theta'].diff()
    dt = df['time'].diff()
    omega = dtheta/dt
    df[r'$\omega$'] = omega
    return df

def spec_dens(acf,columns_in=['$G$']):
    j = acf[columns_in].apply(sp.integrate.simps, x=acf['time'])
    #print(j,acf.iloc[0]['$G$'])
    #tau = j/acf.iloc[0]['$G$']
    return j
