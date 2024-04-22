# This is the standard input file for setting extra paramters needed for initializing and parsing MD trajectories, generating clusters,  

class ParseDynamics:
     MD_ENGINE = 'Tinker'
     traj_dir = './example-data/tinker/methane/'
     #traj_dir = './example-data/tinker/water/vapor/101kpa/' #Path to directory containing trajectory directories {01..XX} ('./example-data/' for example, or './trajectories/' for default space of your own traj data)
     #traj_dir = './example-data/tinker/water/' #Path to directory containing trajectory directories {01..XX} ('./example-data/' for example, or './trajectories/' for default space of your own traj data)
     #ntraj = 1 #number of trajectories to parse
     nat = 1000
     start_prod = 100 #MD step number to start sampling snapshots from. For the default setup and equilibration used in this package, 42000 (~6ps) is recommended
     end_prod = 200
     md_print_freq = 1
     sample_freq = 1 #number of steps between sampled snapshots. Keep high for testing
     celldm = 216.86/0.529177 #Simulation cell dimension in bohr. May be any expression returning floating point value
     #celldm = 16.22/0.529177 #Simulation cell dimension in bohr. May be any expression returning floating point value
     timestep = 0.001 #timestep of MD in picoseconds. May be any expression returning floating point value
     parse_vel = False
class SpinRotation:
    mol_type = 'methane'
    nmol = 100
    #C_SR = [16.495,16.495,-1.875]
    C_SR = [20.23,20.23,20.23] #13C
    sample_freq = 10