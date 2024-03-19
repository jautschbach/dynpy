import numpy as np
import os
import pandas as pd
from itertools import islice
import sys
import getopt
import signal
from dynpy import signal_handler

#signal.signal(signal.SIGINT, signal_handler)

#xyz = "/projects/jochena/adamphil/projects/Na/short/s_07/2_wfcopt/water.xyz_3"
def usage():
    print("USAGE:"+"\n"+
"-h      help"+ "\n"+
"-b      compute boxsize"+"\n"+
"-c      Center an xyz file"+"\n"+
"-r      Randomize analyte"+ "\n"+
"-x      Convert to basic xyz format"+"\n"+
"-q      Write QE input file")

#usage()
def boxsize(fname,dens,deuterated=True):
    #print(dens)
    mw={'H':1.0078, 'D':2.0141, 'C':12.0107, 'N':14.0067, 'O':15.999, 'I':126.90, 'Cs':132.91, 'Xe':131.29, 'Cl':35.45, 'Pt':195.085, 'Na': 22.99}
    aa=6.022e23
    df = pd.read_csv(fname, header=None, skiprows=2, delim_whitespace=True, names=['sym','x','y','z'])

    grouped=df.groupby('sym')
    mass=0
    for group,atoms in grouped:
        symbol = group
        if group == 'H':
            if deuterated:
                symbol = 'D'
        mass+=len(atoms)*mw[symbol]
    #print(mass)
    mass/=aa
    v=(mass/dens)*(1e24)
    #print(v)
    la=v**(1./3.)
    #print(la)
    lb=la/0.529177
    print("Angstrom: "+str(la)+", bohr: "+str(lb))

def toxyz(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    #print(lines)
    l = lines[1]
    #print(line)
    ll = l.split()[1]
    #print(sym)
    try:
        float(ll)
        skip = [0, 1]
    except ValueError:
        skip = [0]
    #print(lines)
    #print(skip)
    #skip=2

    df = pd.read_csv(fname, header=None, skiprows=skip, delim_whitespace=True, names=['A','B','C','D','E','F','G','H','I','J','K'])

    #print(df)

    dfxyz = df.loc[:, ['B','C','D','E']]
    #print(dfxyz)
    syms = dfxyz['B'].tolist()
    newsyms = []
    #print(syms)
    for sym in syms:
        sym=str(sym)
        if len(sym) > 1:
            if((sym[-1].isupper()) or (sym[-1].isnumeric())):
                #print(sym)
                sym = sym[0:len(sym)-1]
        newsyms.append(sym)
    #print(len(newsyms))
    #print(len(dfxyz))
    #print(newsyms)
    dfxyz['B'] = newsyms

    #print(dfxyz.loc[0,1][0])

    #with open(fname + ".og", 'w') as f:
    #    f.write(str(len(dfxyz)))
    #    for row in dfxyz.iterrows():
    #        f.write(str(row))

    dfxyz.to_csv(fname + ".og", sep = ' ', header = False, index = False)

    with open(fname + ".og" ,'r') as f:
        lines = f.readlines()
    with open(fname + ".og" ,'w') as f:
        f.write(str(len(lines))+'\n\n')
        for line in lines:
            f.write(line)

#toxyz(xyz)

def shift(fname,a,r):
    #print(fname,a,r)
    with open(fname, 'r') as f:
        lines = [line for line in islice(f,3) if line!='\n']
    line = lines[1]
    sym = line.split()[1]
    try:
        float(sym)
        skip = 2
        #skip = [0, 1]
    except ValueError:
        skip = 1

    df = pd.read_csv(fname, header=None, skiprows=skip, delim_whitespace=True,names=['A','B','C','D','E','F','G','H','I','J'])
    #print(df)
    if a=='o':
        df['C']-= np.min(df['C'])
        df['D']-= np.min(df['D'])
        df['E']-= np.min(df['E'])
    elif a == 'r':
        r = float(r)
        df['C']+= np.random.random()*r
        df['D']+= np.random.random()*r
        df['E']+= np.random.random()*r
    else:
        a=float(a)
        df['C']+= a
        df['D']+= a
        df['E']+= a

    int_cols = ['F','G','H','I','J']
    df[int_cols] = df[int_cols].fillna(-1)
    #print(df.head())
    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    df[int_cols] = df[int_cols].replace(-1,'')
    #print(df.head())
    df.to_csv(fname + "_s", sep = ' ', header = False, index = False)
    with open(fname + "_s" ,'r') as f:
        lines = f.readlines()
    with open(fname + "_s" ,'w') as f:
        f.write(str(len(lines)) + " molden generated tinker .xyz (oplsaa param.)\n\n")
        for line in lines:
            f.write(line)

#center(xyz)

def toqe(fname, i, system, celldm, ccr_username):

    jobname = str(i).zfill(2)

    with open(fname, 'r') as f:
        lines = f.readlines()

    coord = ''.join(lines[2:])
    nat = len(lines[2:])
    ntyp = len(list(set([l[0] for l in lines[2:]])))

    with open(system+'-'+jobname+'.inp','w') as f:
        f.write("""&control
  calculation = 'cp'
  restart_mode = 'from_scratch'
  pseudo_dir = '/projects/academic/jochena/adamphil/code/pslibrary/revpbe/PSEUDOPOTENTIALS/'
  outdir = '/gpfs/scratch/{4}/{0}/{1}'
  ndr = 50
  ndw = 50
  nstep = 100
  isave = 100
  dt = 2.0
  tprnfor = .true.
  etot_conv_thr = 1.d-10
  ekin_conv_thr = 1.d-10
  prefix = '{0}-{1}'
/
&system
  ibrav = 1
  celldm(1) = {2}
  nat = {5}
  ntyp = {6}
  ecutwfc = 100.0
  nr1b = 25
  nr2b = 25
  nr3b = 25
/
&electrons
  emass = 450.d0
  emass_cutoff = 2.5d0
  ortho_eps = 1.d-10
  ortho_max = 100
  conv_thr = 1.d-10
  electron_dynamics = 'damp'
  electron_velocities = 'zero'
  electron_temperature = 'not_controlled'
/
&ions
  ion_dynamics = 'none'
  ion_velocities = 'zero'
  ion_temperature = 'not_controlled'
/
ATOMIC_SPECIES
I  126.90      I.revpbe-dn-rrkjus_psl.1.0.0.UPF
H  2.01355d0   H.revpbe-rrkjus_psl.1.0.0.UPF
O  15.9994     O.revpbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
{3}""".format(system,jobname,celldm,coord,ccr_username,nat,ntyp))

    with open(system+'-'+jobname+'.inp.2','w') as f:
        f.write("""&control
  calculation = 'cp'
  restart_mode = 'restart'
  pseudo_dir = '/projects/academic/jochena/adamphil/code/pslibrary/revpbe/PSEUDOPOTENTIALS/'
  outdir = '/gpfs/scratch/{4}/{0}/{1}'
  ndr = 50
  ndw = 51
  nstep = 500
  isave = 1000
  dt = 6.0
  tprnfor = .true.
  etot_conv_thr = 1.d-10
  ekin_conv_thr = 1.d-10
  prefix = '{0}-{1}'
/
&system
  ibrav = 1
  celldm(1) = {2}
  nat = {5}
  ntyp = {6}
  ecutwfc = 100.0
  nr1b = 25
  nr2b = 25
  nr3b = 25
/
&electrons
  emass = 450.d0
  emass_cutoff = 2.5d0
  ortho_eps = 1.d-10
  ortho_max = 100
  conv_thr = 1.d-10
  electron_dynamics = 'cg'
  electron_velocities = 'zero'
  electron_temperature = 'not_controlled'
/
&ions
  ion_dynamics = 'none'
  ion_velocities = 'zero'
  ion_temperature = 'not_controlled'
/
ATOMIC_SPECIES
I  126.90      I.revpbe-dn-rrkjus_psl.1.0.0.UPF
H  2.01355d0   H.revpbe-rrkjus_psl.1.0.0.UPF
O  15.9994     O.revpbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
{3}""".format(system,jobname,celldm,coord,ccr_username,nat,ntyp))


    with open(system+'-'+jobname+'.inp.3','w') as f:
        f.write("""&control
  calculation = 'cp'
  restart_mode = 'restart'
  pseudo_dir = '/projects/academic/jochena/adamphil/code/pslibrary/revpbe/PSEUDOPOTENTIALS/'
  outdir = '/gpfs/scratch/{4}/{0}/{1}'
  ndr = 51
  ndw = 52
  nstep = 35000
  isave = 2000
  dt = 6.0
  tprnfor = .true.
  etot_conv_thr = 1.d-10
  ekin_conv_thr = 1.d-10
  prefix = '{0}-{1}'
/
&system
  ibrav = 1
  celldm(1) = {2}
  nat = {5}
  ntyp = {6}
  ecutwfc = 100.0
  nr1b = 25
  nr2b = 25
  nr3b = 25
  vdw_corr = 'dft-d'
/
&electrons
  emass = 450
  emass_cutoff = 2.5d0
  ortho_eps = 1.d-10
  ortho_max = 100
  conv_thr = 1.d-10
  electron_dynamics = 'verlet'
  electron_temperature = 'not_controlled'
/
&ions
  ion_dynamics = 'verlet'
  ion_temperature = 'nose'
  tempw = 300.0
  fnosep = 90.0 45.0 15.0
  nhpcl = 3
  nhptyp = 1
/
ATOMIC_SPECIES
I  126.90      I.revpbe-dn-rrkjus_psl.1.0.0.UPF
H  2.01355d0   H.revpbe-rrkjus_psl.1.0.0.UPF
O  15.9994     O.revpbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
{3}""".format(system,jobname,celldm,coord,ccr_username,nat,ntyp))


    with open(system+'-'+jobname+'.inp.4','w') as f:
        f.write("""&control
   calculation = 'cp'
   restart_mode = 'restart'
   pseudo_dir = '/projects/academic/jochena/adamphil/code/pslibrary/revpbe/PSEUDOPOTENTIALS/'
   outdir = '/gpfs/scratch/{4}/{0}/{1}'
   ndr = 52
   ndw = 53
   nstep = 35000
   isave = 2000
   dt = 6.0
   tprnfor = .true.
   etot_conv_thr = 1.d-10
   ekin_conv_thr = 1.d-10
   prefix = '{0}-{1}'
 /
 &system
   ibrav = 1
   celldm(1) = {2}
   nat = {5}
   ntyp = {6}
   ecutwfc = 100.0
   nr1b = 25
   nr2b = 25
   nr3b = 25
   vdw_corr = 'dft-d'
 /
 &electrons
   emass = 450
   emass_cutoff = 2.5d0
   ortho_eps = 1.d-10
   ortho_max = 100
   conv_thr = 1.d-10
   electron_dynamics = 'verlet'
   electron_temperature = 'not_controlled'
 /
 &ions
   ion_dynamics = 'verlet'
   ion_temperature = 'not_controlled'
   tempw = 300.0
   fnosep = 90.0 45.0 15.0
   nhpcl = 3
   nhptyp = 1
 /
 ATOMIC_SPECIES
 I  126.90      I.revpbe-dn-rrkjus_psl.1.0.0.UPF
 H  2.01355d0   H.revpbe-rrkjus_psl.1.0.0.UPF
 O  15.9994     O.revpbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
{3}""".format(system,jobname,celldm,coord,ccr_username,nat,ntyp))

    with open('md.slm','w') as f:
        f.write("""#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --constraint=IB
#SBATCH --mem=64000
#SBATCH --cluster=industry
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --output={0}-{1}.out
#SBATCH --error={0}-{1}.err
#SBATCH --job-name={0}-{1}

# USER VARS
inp="{0}-{1}.inp"
gpfs="/gpfs/scratch/{2}/{0}/{1}"
execcmd="cp.x"  # QE specific

# MODULES
#module load intel-mpi/5.0.2 mkl/11.2
module load intel-mpi/2017.0.1 mkl/2017.0.1
source /util/academic/intel/17.0/compilers_and_libraries_2017/linux/mpi/intel64/bin/mpivars.sh
source $MKL/bin/mklvars.sh intel64

# ENVIRONMENT VARS
PWSCF="/projects/academic/jochena/adamphil/code/qe-6.0"
NPOOL=1                                         # QE specific
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so   # CCR req'd

# SETUP
mkdir -p ${{gpfs}}

# RUN QE
srun $PWSCF/bin/$execcmd -npool $NPOOL -input $inp
""".format(system,jobname,ccr_username))
    #os.mkdir("/projects/academic/jochena/adamphil/projects/dipolar/acetonitrile/short/" + str(jobname) + "/2-wfcopt/")
    #os.mkdir("/projects/academic/jochena/adamphil/projects/dipolar/acetonitrile/short/" + str(jobname) + "/3-NVT/")
#    with open("/projects/academic/jochena/adamphil/projects/ptcl4/" + str(jobname) + "/2-wfcopt/" + str(jobname)+"-geom.inp", 'w') as f: #s_+ str(jobname) + "/2_wfcopt/" + str(jobname) + ".inp", 'w') as f:
#        f.write("""&control
#  calculation = 'relax'
#  restart_mode = 'from_scratch'
#  pseudo_dir = '/projects/academic/jochena/adamphil/code/pslibrary/revpbe/PSEUDOPOTENTIALS/'
#  outdir = '/gpfs/scratch/adamphil/ptcl4/{0}'
#  tprnfor = .true.
#  disk_io = 'low'
#  etot_conv_thr = 1.d-5
#  forc_conv_thr = 1.d-4
#  prefix = 'ptcl4-{0}'
#  nstep = 500
#/
#&system
#  ibrav = 1
#  celldm(1) = 25.428
#  nat = 199
#  ntyp = 4
#  ecutwfc = 100.0
#  occupations = 'smearing'
#  degauss = 0.01
#/
#&electrons
#  conv_thr = 1.0d-8
#/
#&ions
#  ion_dynamics = 'bfgs'
#/
#ATOMIC_SPECIES
#Pt 195.085 Pt.revpbe-spn-rrkjus_psl.1.0.0.UPF
#Cl 35.45   Cl.revpbe-nl-rrkjus_psl.1.0.0.UPF
#O  15.999  O.revpbe-n-rrkjus_psl.1.0.0.UPF
#H  2.01355 H.revpbe-rrkjus_psl.1.0.0.UPF
#ATOMIC_POSITIONS angstrom
#{1}
#K_POINTS gamma
#""".format(jobname, coord))
#    with open("/projects/academic/jochena/adamphil/projects/dipolar/acetonitrile/short/" + str(jobname) + "/2-wfcopt/" + str(jobname)+".inp", 'w') as f: #s_+ str(jobname) + "/2_wfcopt/" + str(jobname) + ".inp", 'w') as f:
  #      f.write("""&control
  #calculation = 'cp'
  #restart_mode = 'from_scratch'
  #pseudo_dir = '/projects/academic/jochena/alexmarc/code/pseudo'
  #outdir = '/gpfs/scratch/adamphil/dipolar/acetonitrile/short/{0}'
  #ndr = 50
  #ndw = 50
  #nstep = 100
  #isave = 1000
  #dt = 2.0
  #tprnfor = .true.
  #etot_conv_thr = 1.d-10
  #ekin_conv_thr = 1.d-10
  #prefix = 'acetonitrile-{0}'
#/
#&system
  #ibrav = 1
  #celldm(1) = 27.20
  #nat = 192
  #ntyp = 3
  #ecutwfc = 100.0
  #nr1b = 25
  #nr2b = 25
  #nr3b = 25
#/
#&electrons
  #emass = 380.d0
  #emass_cutoff = 2.5d0
  #ortho_eps = 1.d-10
  #ortho_max = 100
  #conv_thr = 1.d-10
  #electron_dynamics = 'sd'
  #electron_velocities = 'zero'
  #electron_temperature = 'not_controlled'
#/
#&ions
  #ion_dynamics = 'none'
  #ion_velocities = 'zero'
  #ion_temperature = 'not_controlled'
#/
#ATOMIC_SPECIES
#C  12.0107     C.pbe-n-rrkjus_psl.1.0.0.UPF
#N  14.0067     N.pbe-n-rrkjus_psl.1.0.0.UPF
#H  1.00784     H.pbe-rrkjus_psl.1.0.0.UPF
#ATOMIC_POSITIONS (angstrom)
#{1}""".format(jobname, coord))

    #with open("/projects/academic/jochena/adamphil/projects/dipolar/acetonitrile/short/" + str(jobname) + "/3-NVT/" + str(jobname)+".inp", 'w') as f: #s_+ str(jobname) + "/2_wfcopt/" + str(jobname) + ".inp", 'w') as f:
        #f.write("""&control
  #calculation = 'cp'
  #restart_mode = 'restart'
  #pseudo_dir = '/projects/academic/jochena/alexmarc/code/pseudo'
  #outdir = '/gpfs/scratch/adamphil/dipolar/acetonitrile/short/{0}'
  #ndr = 51
  #ndw = 52
  #nstep = 35000
  #isave = 1000
  #dt = 6.0
  #tprnfor = .true.
  #etot_conv_thr = 1.d-10
  #ekin_conv_thr = 1.d-10
  #prefix = 'acetonitrile-{0}'
#/
#&system
  #ibrav = 1
  #celldm(1) = 27.20
  #nat = 192
  #ntyp = 3
  #ecutwfc = 100.0
  #nr1b = 25
  #nr2b = 25
  #nr3b = 25
  #vdw_corr = 'dft-d'
#/
#&electrons
  #emass = 380.d0
  #emass_cutoff = 2.5d0
  #ortho_eps = 1.d-10
  #ortho_max = 100
  #conv_thr = 1.d-10
  #electron_dynamics = 'verlet'
  #electron_temperature = 'not_controlled'
#/
#&ions
  #ion_dynamics = 'verlet'
  #ion_temperature = 'nose'
  #tempw = 350.0
  #fnosep = 90.0 45.0 15.0
  #nhpcl = 3
  #nhptyp = 1
#/
#ATOMIC_SPECIES
#C  12.0107     C.pbe-n-rrkjus_psl.1.0.0.UPF
#N  14.0067     N.pbe-n-rrkjus_psl.1.0.0.UPF
#H  1.00784     H.pbe-rrkjus_psl.1.0.0.UPF
#ATOMIC_POSITIONS (angstrom)
#{1}""".format(jobname, coord))

#toqe(xyz, "02")

def rando(xyz,r):
    with open(xyz, 'r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines[1:]):
        ll = line.split()
        lll = [l + np.random.randint(0,r) for l in ll if l in ll[1:4]]
        lines[i] = "   ".join(lll)

    #print(lines)

    with open(xyz + "_r", 'w') as f:
        for line in lines:
            f.write(line)

#rando(xyz)

def xyz_append(xyz1, xyz2):
    with open(xyz1,'r') as f:
        head = f.readlines()[2:]
    with open(xyz2, 'r') as f:
        tail = f.readlines()[2:]
    new = head+tail
    #print(head)
    #print(tail)
    #print(new)
    nat = len(new)
    with open(xyz1+"_app", 'w') as f:
        f.write(str(nat)+"\n\n")
        f.write("".join(new))

def au_to_ang(xyz):
    #print('here')
    df = pd.read_csv(xyz, header=None, skiprows=2, delim_whitespace=True, names=['sym','x','y','z'])
    df[['x','y','z']] = df[['x','y','z']]*0.529177
    df.to_csv("test.xyz", sep=' ', index=False,header=False)
    with open("test.xyz" ,'r') as f:
        lines = f.realines()
    with open("test.xyz" ,'w') as f:
        f.write(str(len(lines))+'\n\n')
        for line in lines:
            f.write(line)

def main():
    try:

        opts, args = getopt.getopt(sys.argv[1:], "hb:s:x:q:e:p:u:")
        #print(opts)
        #print(arg)
    except getopt.GetoptError:
        # Print debug info
        print("no options given")
        usage()
        sys.exit(2)

    #print(opts)

    if opts == []:
        usage()
        sys.exit(2)

    for opt, arg in opts:

        if opt == "-h":
            usage()

        elif opt == "-b":
            dens = float(args[0])
            try:
                deuterated = bool(args[1])
            except IndexError:
                deuterated = True
            boxsize(arg,dens,deuterated)

        elif opt == "-s":
            try:
                a = float(arg)
                r = None
                shift(args[0],a,r)
            except ValueError:
                a = str(arg)
                #print("a = "+a)
                if a == 'r':
                    r = args[0]
                    shift(args[1],a,r)
                elif a == 'o':
                    r = None
                    #print(args)
                    shift(args[0],a,r)


        elif opt == "-x":
            toxyz(arg)


        elif opt == "-q":
            #print(args)
            celldm = float(args[2])/0.529177
            toqe(arg, args[0], args[1], celldm, args[3])

        elif opt == "-e":
            xyz = str(arg)

        elif opt == "-p":
            xyz_append(arg,xyz)

        elif opt == "-u":
            au_to_ang(arg)

main()
