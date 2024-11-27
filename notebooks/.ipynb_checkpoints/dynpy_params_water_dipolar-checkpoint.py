#This is a standard input file for setting parameters needed for parsing MD trajectories, and calculating dipolar 1H relaxation rates. Current values are chosen for the Dipolar/Tinker example data set.

class ParseDynamics:
     MD_format = 'Tinker' #Format of MD trajectory file to parse. Can be 'QE','Tinker','xyz'
     traj_dir = '../dynpy-examples/Dipolar/Tinker/' #Path to directory containing trajectory directories {01..XX}
     trajs = ['01'] # List of trajectory directory names to parse as MD trajectories. Must have the format {'01'..'XX'}
     nat = 384 #Number of atoms per frame
     start_prod = 100 #Step number to start sampling snapshots from. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
     end_prod = 400 #Step number to stop sampling snapshots. Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different. Inclusive, 1-indexed
     md_print_freq = 50 #How many MD timesteps between prints in MD output. Needed for Tinker and xyz format output where MD frames are not explicitly printed
     sample_freq = 10 #number of steps between sampled snapshots.Enumerates the PRINTED FRAMES ONLY, not the original MD frame numbers if they are different.
     celldm = 16.22/0.529177 #Simulation cell dimension in bohr. May be any expression returning floating point value
     timestep = 0.001 #timestep of MD in picoseconds. May be any expression returning floating point value
class Dipolar:
    identifier = 'H(2)O(1)' #Molecular formula unit for molecule of interest in the format shown here
    pair_type = 'all' #'intra','inter',or 'all' for intramolecular or intermolecular contributions separately or both simultaneously ('all')
    rmax = 15 #Max distance in bohr for tracking 2-atom distances
