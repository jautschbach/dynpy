# This is a standard input file for setting parameters needed for parsing MD trajectories, generating clusters, and input files for EFG calculations. Current values are chosen for the Quadrupolar example data set.

class ParseDynamics:
    MD_format = 'QE'  #Format of MD trajectory file to parse. Can be 'QE','Tinker','xyz'
    traj_dir = '../dynpy-examples/Quadrupolar/' #Path to directory containing trajectory directories {01..XX}
    trajs = ['01'] # List of trajectory directory names to parse as MD trajectories. Must have the format {'01'..'XX'}
    start_prod = 4200 #Step number to start sampling snapshots from. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
    end_prod = 4240  #Step number to stop sampling snapshots. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
    sample_freq = 20 #number of steps between sampled snapshots.Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different.
    md_print_freq = 10 #How many MD timesteps between prints in MD output. Needed for Tinker and xyz format output where MD frames are not explicitly printed
    timestep = 6.0*2.418884254e-5 #timestep of MD in picoseconds. May be any expression returning floating point value
    nat = 194  #Number of atoms per frame
    symbols = ['I'] + ['H']*129 + ['O']*64 #Full list of element symbols in the order they appear in 'ATOMIC_SPECIES' block of QE MD input
    celldm = 24.269 #Simulation cell dimension in bohr. May be any expression returning floating point value

class Snapshots:
    write_ADF = True  #True if you want to write ADF inputs to calculate EFGs
    write_GIPAW = True  #True if you want to write QE-PAW inputs to calculate EFGs
    write_xyzs = True  #True if you want to generate a set of xyzs for the snapshots
    skip_compute_neighbors = False #True if you have already generated xyzs of clusters and want to generate ADF inputs
    analyte_symbol = 'I' #Symbol of analyte atom to form clusters around
    analyte_label = 'I'  # Can be same as analyte symbol if it is unique in the system. Otherwise, provide integer index of analyte atom (0-indexed)
    nn = 30  #number of nearest neighbor molecules for clustering
    solute_charge = -1 #Charge of solute atom for computing overall charge of clusters
    formal_charges = {'O':-2,'H':1} #Formal charges of other atoms in the system for computing overall charge of clusters
    scratch = "/scratch/space/for/EFG/calculations/" #Path for scratch space to use if running EFG calculations

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
