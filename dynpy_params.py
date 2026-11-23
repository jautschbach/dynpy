# This is the standard input file for setting extra paramters needed for parsing MD trajectories, generating clusters, writing inputs for snapshot properties calculations, parsing the output of those calculations, and finally calculating relaxation rates. This file is not for any particular calculation but rather serves as an example with a description provided for each input parameter.

class ParseDynamics:
    MD_ENGINE = 'QE' #Indicates the format of the MD trajectory data. Supported values are 'QE', 'Tinker', and 'xyz'.
    traj_dir = './' #Path to directory containing trajectory directories {01..XX} ('./example-data/' for examples)
    trajs = ['01'] # List of trajectory directory names to parse as MD trajectories. Must have the format {'01'..'XX'}
    start_prod = 4200 #MD step number to start sampling snapshots from. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
    end_prod = 4240  #MD step number to stop sampling snapshots. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
    sample_freq = 20 #number of steps between sampled snapshots.Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different.
    md_print_freq = 10 #How many MD timesteps between prints in MD output. Needed for Tinker and xyz format output where MD frames are not explicitly printed
    timestep = 6.0*2.418884254e-5 #timestep of MD in picoseconds. May be any expression returning floating point value
    nat = 194  #Number of atoms per frame
    symbols = ['I'] + ['H']*129 + ['O']*64 #Full list of element symbols in the order they appear in 'ATOMIC_SPECIES' block of QE MD input
    celldm = 24.269 #Simulation cell dimension in bohr. May be any expression returning floating point value
    

class Snapshots:  #Input parameters file for neighbors.py. Used for parsing QE aiMD, making clusters, and writing inputs for ADF/QE-GIPAW
    write_ADF = True #If True, write input files for ADF EFG calcs
    write_GIPAW = False #If True, write input files for QE-PAW EFG calcs
    analyte_label = 'I' #Element symbol for analyte nucleus. Should be redundant with nuc_symbol and if not provided will be inferred from the latter
    nuc_symbol = '127I' #Analyte symbol prepended with the proper atomic mass number of the isotope
    solute_charge = -1 #Charge of the solute/analyte
    formal_charges = {'H':2,'O':-2} #Formal charges of other atoms in the system for computing overall charge of clusters
    write_xyzs = True #If True, write set of xyz files of clusters used for ADF inputs
    nn = 30 #Number of nearest neighbor solvent molecules desired for clusters
    scratch = "/scratch/space/for/EFG/calculations/" #Path for scratch space to use if running EFG calculations
    skip_compute_neighbors = False #If True, write new ADF inputs from existing xyz data from previous run. Use only if .xyz files exist and you know they are computed correctly

class ParseEFGs:
    output_dir = './pre-computed-EFGs/' #Path to directory containing trajectory directories which must be named in the format {01..XX}
    engine = 'ADF' #QM Engine that EFG outputs come from. Can be 'ADF','QE','CP2K'
    prefix = 'I-ADF' # Simple prefix for naming output EFG data file
    time_bw_frames = None #Optional parameter to provide the real-time step between the sampled frames. If not present or set to None, will attempt to read from comment in EFG input/output

class SpinRotation:
    mol_type = 'methane' #Molecule type for which to compute 1H SR relaxation rate. Supported types are 'methane', 'water', 'acetonitrile', and 'methyl' for methyl groups.
    nmol = 100 #Number of molecules to sample when computing angular velocities, correlation functions, and SR rate. Can be up to total number of molecules in the system but may be reduced for performance.
    C_SR = [16.495,16.495,-1.875] #Principal components of SR tensor in the molecule-fixed coordinate system. Current values are for 1H in free methane. Units are kHz
    #C_SR = [20.23,20.23,20.23] # 13C in methane
    #C_SR = [36.9, 33.5, 35.5] #1H in water
    #C_SR = [0.48124,0.48124,15.9531] #1H in acetonitrile


class Quadrupolar:
    data_set = './I-ADF-efg.csv'
    analyte = '127I'
    multiple_trajectories = False
    timestep = 0.01015931
    index_is_frame = True


class Dipolar:
    identifier = 'H(2)O(1)' #Molecular formula unit for molecule of interest in the format shown here
    pair_type = 'all' #'intra','inter',or 'all' for intramolecular or intermolecular contributions separately or both simultaneously ('all')
    rmax = 15 #Max distance in bohr for tracking 2-atom distances


#In most cases, the following templates do not need to be changed
class InputTemplates:  #Templates for ADF inputs
    ADF_in = """ATOMS
{0}
END

RELATIVISTIC Scalar ZORA

BASIS
core None
{3} ZORA/QZ4P/{3}
O ZORA/DZP/O
H ZORA/DZP/H
END

XC
HYBRID PBE0
END

CHARGE {2}

SOLVATION
solv eps=78.54 rad=1.93
END

QTENS

SYMMETRY nosym

NUMERICALQUALITY verygood

COMMENT
{1}
END

SAVE TAPE21 TAPE10"""

#Template for ADF Slurm scripts
    ADF_slm = """#!/usr/bin/env bash
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=64000
##SBATCH --exclusive
##SBATCH --constraint=IB
#SBATCH --cluster=industry
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --job-name=ADF/{0}/{1}
#SBATCH --output={1}.out
#SBATCH --error={1}.err

prefix={1}
scratch={2}/$SLURM_JOB_NAME/
mkdir -p $scratch

source 
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export SCM_TMPDIR=$scratch
export SLURMTMPDIR=$scratch

cd $scratch
cp $SLURM_SUBMIT_DIR/$prefix*.inp .
$ADFBIN/adf < $prefix-scf.inp > $prefix-scf.out
cp $prefix*.out $SLURM_SUBMIT_DIR/"""

    #Template for QE-GIPAW scf input
    PAW_scf_in = """&control
!{0}
calculation = 'scf'
prefix = '{1}'
restart_mode = 'from_scratch'
pseudo_dir = ''
outdir = './'
verbosity = 'low'
wf_collect = .true.
/
&system
ibrav = 1
celldm(1) = {2}
nat = {3}
ntyp = {4}
ecutwfc = 100
/
&electrons
electron_maxstep = 200
conv_thr =  1e-7
/
ATOMIC_SPECIES
I  126.90      I.revpbe-dn-kjpaw_psl.1.0.0.UPF
H  2.01355d0   H.revpbe-kjpaw_psl.1.0.0.UPF
O  15.9994     O.revpbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS (angstrom)
{5}
K_POINTS automatic
1 1 1 0 0 0"""

    #Template for QE-GIPAW efg input
    PAW_efg_in = """&inputgipaw
!{0}
job = 'efg'
prefix = '{1}'
tmp_dir = './'
verbosity = high
spline_ps = .true.
Q_efg(1) = {2}
/"""

#Template for QE-GIPAW Slurm scripts
PAW_slm = """#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cluster=faculty
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
##SBATCH --exclusive
#SBATCH --output={0}.out
#SBATCH --error={0}.err
#SBATCH --job-name=GIPAW/{1}/{0}
#SBATCH --mem=187000

# USER VARS
gpfs={2}/$SLURM_JOB_NAME/

#GET INPUT FROM SUBMIT DIR
cd $SLURMTMPDIR
cp $SLURM_SUBMIT_DIR/*.inp $SLURMTMPDIR

# MODULES
module load intel
module load intel-mpi
module load mkl

# ENVIRONMENT VARS
PWSCF=""
GIPAW=""
NPOOL=1                                         # QE specific
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so   # CCR req'd

# SETUP
mkdir -p ${{gpfs}}

# RUN QE
srun $PWSCF/bin/pw.x -npool $NPOOL -input {0}-scf.inp > {0}-scf.out
srun $GIPAW/bin/gipaw.x -npool $NPOOL -input {0}-efg.inp > {0}-efg.out

# SAVE FILES
cp {0}-{{scf,efg}}.out $SLURM_SUBMIT_DIR
cp -r * ${{gpfs}}"""
