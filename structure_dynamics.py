import os
import exa
import exatomic
from exatomic import qe
from exatomic.algorithms import diffusion
from exatomic.algorithms import pcf

def parse_qe(traj_dir,symbols,celldm,start_prod=42000,sample_freq=400,compute_distances=True):
    unis = {}
    trajs = [t.name for t in os.scandir(traj_dir) if t.name.isnumeric()]
    print("Parsing "+ str(len(trajs)) + " trajectories...")
    for t,traj in enumerate(trajs):
        print("Trajectory "+traj+'...')
        pos_dir = traj_dir + traj + '/'
        xyzs = pos_dir + "xyzs/"
        pos = list(filter(lambda x: ".pos" in x, os.listdir(pos_dir)))[0]
        atom = qe.parse_xyz(pos_dir+'/'+pos,symbols=symbols)
        atom['frame'] = atom['frame'].astype(int)
        atom = atom[(atom['frame']>=atom.frame[0] + start_prod) & (atom['frame']%sample_freq==0)]

        u = exatomic.Universe()
        u.atom = atom
        u.atom['label'] = u.atom.get_atom_labels().astype(int)
        
        # Add the unit cell dimensions to the frame table of the universe
        for i, q in enumerate(("x", "y", "z")):
            for j, r in enumerate(("i", "j", "k")):
                if i == j:
                    u.frame[q+r] = celldm
                else:
                    u.frame[q+r] = 0.0
            u.frame["o"+q] = 0.0
        u.frame['periodic'] = True

        if compute_distances:
            u.compute_atom_two(vector=True,dmax=15)
            u.compute_molecule()
            u.atom['molecule_label']=u.atom[u.atom['frame']==u.atom.iloc[0,3]].molecule.values.tolist()*len(u.atom.frame.unique())
            u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame'])
            u.atom_two.loc[:,'label0'] = u.atom_two.atom0.map(u.atom['label'])
            u.atom_two.loc[:,'label1'] = u.atom_two.atom1.map(u.atom['label'])
            u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
            u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
            u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule_label'])
            u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule_label'])
    unis[t] = u
    print("Done.")
    return unis

def RDF(unis,sym1,sym2,start=1,stop=13):
    rdfs = {t:pcf.radial_pair_correlation(u,sym1,sym2,start,stop) for (t,u) in unis.items()}
    return rdfs

def coodination_numbers(unis,rcut,sym1,sym2):
    Shells={}
    for (t,u) in unis.items():
        two = u.atom_two
        Itwo = two[(two['symbol0']==sym1) & (two['symbol1']==sym2)]
        close = Itwo[Itwo['dr']<=rcut/0.529177]
        shells = pd.DataFrame(close.groupby('frame')['molecule1'].apply(pd.unique))
        shells.loc[:,'n'] = shells['molecule1'].apply(len)
        Shells[t] = shells

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

#Vector autocorrelation
def vector_corr(V,rotational=False,normalize=True):
    N = len(V)
    a = np.array(V)
    c1 = np.empty(N)
    c2 = np.empty(N)
    for t in range(N):
        #print(t)
        Sum1 = 0
        Sum2 = 0
        for n in range(N-t):
            Sum1 += np.vdot(a[n],a[n+t])
            #print(a[n], a[n+t])
            if rotational:
                Sum2 += (1/2)*(3*(np.vdot(a[n],a[n+t])**2) - 1)
        c1[t] = Sum1/N
        c2[t] = Sum2/N
    if rotational:
        if normalize:
            return c1/c1[0],c2/c2[0]
        else:
            return c1,c2
    elif normalize:
        return c1/c1[0]
    else:
        return c1

def plane_norm(df):
    try:
        a = np.array(df.iloc[0][['dx','dy','dz']].astype(float))
        b = np.array(df.iloc[1][['dx','dy','dz']].astype(float))
        return np.cross(a,b)
    except IndexError:
        return np.array([0,0,0])

def rotational_correlation(unis,timestep=6.0,compute_distances=True):
    dt = timestep*2.418884e-5
    if compute_distances:
        u.compute_atom_two(vector=True,dmax=5)
        u.compute_molecule()
        u.atom['molecule_label']=u.atom[u.atom['frame']==u.atom.iloc[0,3]].molecule.values.tolist()*len(u.atom.frame.unique())
        u.atom_two.loc[:,'frame'] = u.atom_two.atom0.map(u.atom['frame'])
        u.atom_two.loc[:,'label0'] = u.atom_two.atom0.map(u.atom['label'])
        u.atom_two.loc[:,'label1'] = u.atom_two.atom1.map(u.atom['label'])
        u.atom_two.loc[:,'symbol0'] = u.atom_two.atom0.map(u.atom['symbol'])
        u.atom_two.loc[:,'symbol1'] = u.atom_two.atom1.map(u.atom['symbol'])
        u.atom_two.loc[:,'molecule0'] = u.atom_two.atom0.map(u.atom['molecule_label'])
        u.atom_two.loc[:,'molecule1'] = u.atom_two.atom1.map(u.atom['molecule_label'])
    for t,u in unis.items():
        two = u.atom_two
        waters = two[(two['bond']==True) & (two['symbol0']=='O') & (two['symbol1']=='H')]

        grouped = waters.groupby(("frame","molecule0"))

        normals = grouped.apply(plane_norm)

        racfs = {}
        grouped = normals.groupby("molecule0")
        for molecule, group in grouped:
            racf = vector_corr(group.values,rotational=True,normalize=True)
            racfs[molecule] = racf[1]

        RACFs = pd.DataFrame(racfs)

        RACF = pd.DataFrame(RACFs.apply(np.mean,axis=1))

        RACF['frame'] = waters.frame.unique().astype(int)
        RACF['time'] = RACF['frame']*dt
        RACF.rename(columns={0:"RACF"},inplace=True)
        RACF['time'] = RACF['time']-RACF['time'].min()
        RACFS[t] = RACF
    return(RACFS)
