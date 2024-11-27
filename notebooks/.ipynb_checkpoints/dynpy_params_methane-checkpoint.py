#This is a standard input file for setting parameters needed for parsing MD trajectories, and calculating Spin-Rotation 1H relaxation rates. Current values are chosen for the Spin-rotation/methane example data set.

class ParseDynamics:
     MD_format= 'Tinker' #Format of MD trajectory file to parse. Can be 'QE','Tinker','xyz'
     traj_dir = './' #Path to directory containing trajectory directories {01..XX}
     trajs = ['01'] #List of trajectory directory names to parse as MD trajectories. Must have the format {'01'..'XX'}
     nat = 1000 #Number of atoms per frame
     start_prod = 100 #Step number to start sampling snapshots from. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
     end_prod = 1000 #Step number to stop sampling snapshots. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
     md_print_freq = 1 #How many MD timesteps between prints in MD output. Needed for Tinker and xyz format output where MD frames are not explicitly printed
     sample_freq = 10 #Number of steps between sampled snapshots. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different.
     celldm = 216.86/0.529177 #Simulation cell dimension in bohr. May be any expression returning floating point value
     timestep = 0.001 #timestep of MD in picoseconds. May be any expression returning floating point value
     parse_vel = False #Whether or not to parse a .vel file for explicit velocities. If false and running SR calc, will attempt to calculate based on positions and timestep. This may be inaccurate if sampled timestep is too long.
class SpinRotation:
    mol_type = 'methane' #Molecule type for which to compute 1H SR relaxation rate. Supported types are 'methane', 'water', 'acetonitrile', and 'methyl' for methyl groups.
    nmol = 100 #Number of molecules to sample when computing angular velocities, correlation functions, and SR rate. Can be up to total number of molecules in the system but may be reduced for performance.
    C_SR = [16.495,16.495,-1.875] #Principal components of SR tensor in the molecule-fixed coordinate system. Current values are for 1H in free methane. Units are kHz
    #C_SR = [20.23,20.23,20.23] # 13C in methane
    #C_SR = [36.9, 33.5, 35.5] #1H in water
    #C_SR = [0.48124,0.48124,15.9531] #1H in acetonitrile
