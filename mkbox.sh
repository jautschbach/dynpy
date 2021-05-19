#!/bin/bash

#Define some option variables
system=I #Simple description of system for naming files
ntrajectories=1 #number of boxes to initialize
solvent=water #name of coordinate file of solvent
solute=I #name of coordinate file of solute
counterion=hydronium #name of coordinate file of counterion
nsolv=63 #Number of solvent molecules
cell_dm=12.838 #Box size (angstrom)
r_max=12 #parameter for random placement of counterion. Can not be larger than cell_dm
ccr_username=adamphil #Pointers for pseudopotential files and for saving scratch data assume CCR conventions

#Pack for X number of independent trajectories {01..XX}
for i in $(seq -w 01 $ntrajectories);
do
rm -r $i;
mkdir $i;
mkdir $i/packing;
cd $i/packing;

#tinker coorrdinate files (.txyz) and .key must be in current working directory
cp ../../${solvent}.txyz .;
cp ../../${solute}.txyz .;
cp ../../${counterion}.txyz .;
cp ../../tinker.key .;

#Initialize box and pack with water. Can use rough estimate for box dimensions (angstrom), then use python -m tinkertoys -b ${solvent}.xyz <density> to calculte more precicely.
echo -e "${solvent}.txyz\n19\n$nsolv\n$cell_dm $cell_dm $cell_dm\nY" | xyzedit -k tinker.key;

#Quick optimization
echo -e "${solvent}.xyz\n0.1" | minimize -k tinker.key;

#Shift box to origin
python -m tinkertoys -a o -s ${solvent}.xyz_2;

#Add solute to box by appending solute coordinates. Solute.txyz coordinates should roughly correspond to the center of the box
echo -e "${solvent}.xyz_2_s\n18\n${solute}.txyz" | xyzedit -k tinker.key;

#randomize counterion position 0 < x,y,z < 12. Max defined by -r input variable
python -m tinkertoys -a r -r 12 -s ${counterion}.txyz

#Append counterion coordinates to box
echo -e "${solvent}.xyz_3\n18\n${counterion}.txyz_s" | xyzedit -k tinker.key;

#Optimize
echo -e "${solvent}.xyz_4\n0.02" | minimize -k tinker.key;

#Classical NVT: 1000 steps, 1fs timestep, print interval 1ps, 2 indicates NVT ensemble, temperature=300K
echo -e "${solvent}.xyz_5\n1000\n1\n1\n2\n300" | dynamic -k tinker.key;

#Center box on origin
python -m tinkertoys -a o -s ${solvent}.arc;

#Convert to standard .xyz format
python -m tinkertoys -x ${solvent}.arc_s;

#Write qe inputs and .slm. Manually edit string in toqe() function in tinkertoys.py or generated input files if needed. Writes first four steps of aimd: initial wf opt (*.inp and *inp.2), NVT heating (*.inp.3), and first 5ps NVE production (*inp.4)
python -m tinkertoys -q ${solvent}.arc_s.og ${i} $system $cell_dm $ccr_username;
mv *inp* ..;
mv md.slm ..;

cd -;

done
