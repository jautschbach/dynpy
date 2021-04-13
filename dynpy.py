import exa
exa.logging.disable(level=10)
exa.logging.disable(level=20)
import sys
import getopt
# import pandas as pd
# import numpy as np
# import scipy as sp
# import os
# import datetime as dtt
# import string
# from numpy import linalg as la
# import exa
# import exatomic
# from exatomic import qe
# from neighbors_input import *
# from exatomic.algorithms import neighbors
# from exatomic.algorithms.neighbors import periodic_nearest_neighbors_by_atom    # Only valid for simple cubic periodic cells
# from nuc import *
from neighbors_input import *
from neighbors import *
from parse import *
from qrax import *

def usage():
    print("USAGE:"+"\n"+
    "-h      --help                                                                       Displays this message"+ "\n"+
    "-i      --inputs                                                                     Generate EFG calculation inputs for ADF and/or QE-GIPAW. Requires parameters defined in neighbors_input.py"+"\n"+
    "-e      --parse-efgs <ADF/GIPAW> <path to trajectories> <system/calc description>    Parse EFG data from ADF or GIPAW outputs. Last two arguments are optional. Defaults: \"./example-data/\", <ADF/GIPAW>"+ "\n"+
    "-r      --relax <efg data file> <analyte symbol> <numeric analyte label>             Compute relaxation rates and related parameters. Analyte symbol is mass number followed by element symbol e.g. 127I." + "\n" +
    "                                                                                      Last argument is optional (use if you know atom labels are consistent in efg data and you want to designate analyte by numeric label)")

def main(traj_dir=traj_dir):
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hie:r:",["help",'inputs','parse-efgs=','relax='])
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
            gen_inputs(traj_dir)
        elif opt in ("-e","--parse-efgs"):
            try:
                traj_dir = args[0]
                system = args[1]
            except IndexError:
                if len(args) == 0:
                    print("\nPath to trajectories and calculation description not provided. Defaults are \"./example-data/\" and \""+arg+"\" respectively\n")
                    traj_dir = "./example-data/"
                    system = arg
                else:
                    Print("\nPlease provide BOTH path to trajectories and calculation description, OR NEITHER in order to use defaults\n")
                    usage()
                    sys.exit(2)
            if arg == "ADF":
                extract_efg_adf(system,traj_dir)
            elif arg == "GIPAW":
                extract_efg_qe(system,traj_dir)
            else:
                usage()
                sys.exit(2)
        elif opt in ("-r","--relax"):
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
            Q_rax(arg,analyte,label)

if __name__ == "__main__":
    main()
