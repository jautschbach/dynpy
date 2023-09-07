import sys
import getopt
import inspect

def usage():
    print("USAGE:"+"\n"+
    "-h   --help                                Displays this message"+ "\n\n"+
    "-i   --inputs                              Generate EFG calculation inputs for ADF and/or QE-GIPAW. Requires parameters defined in neighbors_input.py"+"\n\n"+
    "-e   --parse-efgs                          Parse EFG data from ADF or GIPAW outputs."+ "\n\n"+
    "-q   --Qrelax                              Compute quadrupolar relaxation rates and related parameters.\n\n"+
    "-s   --SRrelax                             Calculate 1H SR relaxation rate from MD trajectory.\n"+
    "                                           Currently supported molecule types are 'water','acetonitrile', and 'methane'\n\n"+
    "-d   --DDrelax")

def read_input(input_arg, required):
    try:
        file = input_arg.split('.py')[0].strip('./\\')
    except:
        print("\nPlease provide, as an argument, an input file with required paramaters. See dynpy_params.py\n")
        sys.exit(2)
    try:
        dynpy_params = __import__(file)
    except ImportError:
        print("\nInput parameter file must have .py file extension and have valid python syntax. See dynpy_params.py\n")
        sys.exit(2)
    input_classes = inspect.getmembers(dynpy_params,inspect.isclass)
    all_input_vars = [varr for clas in input_classes for varr in list(clas[1].__dict__.keys()) if "__" not in varr]
    for clas, reqs in required.items():
        for req in reqs:
            if req not in all_input_vars:
                print("Missing required input variable %s in class %s. See dynpy_params.py" % (req,clas))
                sys.exit(2)
    return dynpy_params


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:q:s:d:",["help",'inputs','parse-efgs','Qrelax','SRrelax','DDrelax'])
    except getopt.GetoptError:
        # Print debug info
        usage()
        sys.exit(2)

    #print(opts)

    if opts == []:
        usage()
        sys.exit(2)

    for opt, arg in opts:

        if opt in ("-h","--help"):
            usage()

        elif opt in ("-i","--inputs"):
            required = {'ParseDynamics': ['MD_ENGINE','traj_dir','sample_freq','timestep','celldm'],
                        'Snapshots': ['write_ADF','write_GIPAW','nuc_symbol']}
            dynpy_params = read_input(args[0],required)
            
            import neighbors
            neighbors.gen_inputs(dynpy_params)

        elif opt in ("-e","--parse-efgs"):
            import parseEFG
            try:
                traj_dir = args[0]
                system = args[1]
            except IndexError:
                    print("\nInsufficient arguments provided\n")
                    usage()
                    sys.exit(2)
            try:
                time_bw_frames = float(args[2])
            except:
                time_bw_frames = None
            if arg == "ADF":
                parseEFG.extract_efg_adf(traj_dir,system)
            elif arg == "GIPAW":
                parseEFG.extract_efg_qe(traj_dir,system,time_bw_frames)
            elif arg == "CP2K":
                parseEFG.extract_efg_cp2k(traj_dir,system,time_bw_frames)
            else:
                usage()
                sys.exit(2)
        elif opt in ("-q","--Qrelax"):
            import qrax
            try:
                if args[0].isnumeric() or args[0].isalpha():
                    print("\nMust provide designation of analyte nucleus as the mass number followed by element symbol e.g. 127I\n")
                    usage()
                    sys.exit(2)
                else:
                    analyte = args[0]
            except IndexError:
                print("\nMust provide designation of analyte nucleus as the mass number followed by element symbol e.g. 127I\n")
                usage()
                sys.exit(2)
            try:
                label = args[1]
            except IndexError:
                label = None
            qrax.Q_rax(arg,analyte,label)
        
        elif opt in ("-s","--SRrelax"):
            required = {'ParseDynamics': ['MD_ENGINE','traj_dir','sample_freq','timestep','celldm'],
                        'SpinRotation': ['mol_type','C_SR']}
            dynpy_params = read_input(args[0],required)
            import SRparse

            SRparse.SpinRotation(dynpy_params)

        elif opt in ("-d","--DDrelax"):
            import ddrax
            ddrax.DD_func()

if __name__ == "__main__":
    main()
