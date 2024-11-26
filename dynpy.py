import sys
import os
import getopt
import inspect
import signal as sig

def usage():
    print("USAGE:"+"\n"+
    "-h   --help                                Displays this message"+ "\n\n"+
    "-i   --inputs <params.py>                  Generate EFG calculation inputs for ADF and/or QE-PAW."+"\n\n"+
    "-e   --parse-efgs <params.py>              Parse EFG data from ADF or GIPAW outputs."+ "\n\n"+
    "-q   --Qrelax <params.py>                  Compute quadrupolar relaxation rates and related parameters from EFG time series.\n\n"+
    "-s   --SRrelax <params.py>                 Calculate SR relaxation rates and related parameters from MD trajectory.\n"+
    "                                               Currently supported molecule types are 'water','acetonitrile', 'methane', and methyl groups\n\n"+
    "-d   --DDrelax <params.py>                 Compute dipolar relaxation rates and related parameters from MD trajectory.")

def read_input(input_arg, required):
    try:
        file = input_arg.split('.py')[0].strip('./\\')
    except:
        print("\nPlease provide, as an argument, an input file with required paramaters.\n")
        sys.exit(2)
    try:
        dynpy_params = __import__(file)
    except ImportError:
        print("\nInput parameter file must have .py file extension and have valid python syntax.\n")
        sys.exit(2)
    input_classes = inspect.getmembers(dynpy_params,inspect.isclass)
    all_input_vars = [varr for clas in input_classes for varr in list(clas[1].__dict__.keys()) if "__" not in varr]
    for clas, reqs in required.items():
        for req in reqs:
            if req not in all_input_vars:
                print("Error: Missing required input variable %s in class %s." % (req,clas))
                sys.exit(2)
    return dynpy_params

#def signal_handler(sig, frame):
#    print('Keyboard Interrupt')
#    sys.exit(2)


def main():
    #signal.signal(signal.SIGINT, signal_handler)
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:e:q:s:d:",["help",'inputs=','parse-efgs=','Qrelax=','SRrelax=','DDrelax='])
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
            required = {'ParseDynamics': ['MD_format','traj_dir','timestep','celldm','trajs','md_print_freq','nat','start_prod','end_prod'],
                        'Snapshots': ['write_ADF','write_GIPAW']}
            dynpy_params = read_input(arg,required)
            
            import neighbors
            neighbors.gen_inputs(dynpy_params)

        elif opt in ("-e","--parse-efgs"):
            required = {'ParseEFGs': ['output_dir','engine']}
            dynpy_params = read_input(arg,required)
            PE = dynpy_params.ParseEFGs
            import parseEFG
            if  PE.engine == "ADF":
                parseEFG.extract_efg_adf(PE.output_dir,PE.prefix)
            elif PE.engine == "QE":
                parseEFG.extract_efg_qe(PE.output_dir,PE.prefix,PE.time_bw_frames)
            elif PE.engine == "CP2K":
                parseEFG.extract_efg_cp2k(PE.output_dir,PE.prefix,PE.time_bw_frames)
            else:
                print("EFG engine must be ADF, QE, or CP2K")
                sys.exit(2)

        elif opt in ("-q","--Qrelax"):
            required = {'Quadrupolar': ['data_set','analyte']}
            dynpy_params = read_input(arg,required)
            QR = dynpy_params.Quadrupolar
            import qrax
            # try:
            #     if args[0].isnumeric() or args[0].isalpha():
            #         print("\nMust provide designation of analyte nucleus as the mass number followed by element symbol e.g. 127I\n")
            #         usage()
            #         sys.exit(2)
            #     else:
            #         analyte = args[0]
            # except IndexError:
            #     print("\nMust provide designation of analyte nucleus as the mass number followed by element symbol e.g. 127I\n")
            #     usage()
            #     sys.exit(2)
            # try:
            #     label = args[1]
            # except IndexError:
            #     label = None
            qrax.QR_module_main(QR)
        
        elif opt in ("-s","--SRrelax"):
            required = {'ParseDynamics': ['MD_format','traj_dir','timestep','celldm','trajs','md_print_freq','nat','start_prod','end_prod'],
                        'SpinRotation': ['mol_type','C_SR']}
            dynpy_params = read_input(arg,required)
            PD = dynpy_params.ParseDynamics
            SR = dynpy_params.SpinRotation
            import SRparse
            SRparse.SR_module_main(PD,SR)

        
        elif opt in ("-d","--DDrelax"):
            required = {'ParseDynamics': ['MD_format','traj_dir','timestep','celldm','trajs','md_print_freq','nat','start_prod','end_prod'],
                        'Dipolar':['identifier']}
            dynpy_params = read_input(arg,required)
            PD = dynpy_params.ParseDynamics
            DD = dynpy_params.Dipolar
            import DDparse
            DDparse.DD_module_main(PD,DD)

if __name__ == "__main__":
    main()
