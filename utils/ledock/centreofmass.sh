#!/bin/bash

# This bash script is used to generate ligand docking poses for 
# ligands in PDBbind v2020 general and refined set. Please make
# sure that the centre of mass of each ligand in these sets has
# been calculated via centreofmass.ipynb. The outputs should be
# in .txt format with space delimiter. 

# By default, we set the number of docking poses generated to be
# 10 for each complex and the RMSD=1.0. You may wish to redefine 
# any parameter in LeDock's INPUT file. Also, we assume that you
# have the same directory structure as mentioned on our GitHub's
# repository.

# USAGE: nohup bash ./centreofmass.sh -r 2.0 -n 20 -s 10 &
# ARGUMENTS:
# -r: RMSD for clustering as in LeDock's requirement;
# -n: Number of binding poses needed;
# -s: Spacing/dimension of the binding pocket.
# Check LeDock's documentation for more info.
# As the process takes a long time, we recommend running in bg.


LEDOCK=./LeDock
GENERAL_FILE="./general-set-com.txt"
REFINED_FILE="./refined-set-com.txt"

# Parse command-line arguments using getopts
while getopts "r:n:s:" flag; do
    case $flag in
        r)
            rmsd="$OPTARG"
            ;;
        n)
            num="$OPTARG"
            ;;
        s)
            spacing="$OPTARG"
            ;;
        *)
            echo "Usage: $0 -r <rmsd> -n <num> -s<spacing>"
            exit 1
            ;;
    esac
done

# Check if both rmsd and num were provided
if [[ -z $rmsd ]] || [[ -z $num ]] || [[ -z $spacing ]]; then
    echo "ALL -r and -n and -s flags are required."
    echo "Usage: $0 -r <rmsd> -n <num> -s <spacing>"
    exit 1
fi

# Use the provided values
echo "The value of rmsd is: $rmsd"
echo "The value of num is: $num"
echo "The value of spacing is: $spacing"
echo "================================="

# Count how many complexes available for docking, including the last line
num_lines=$( wc -l < ${GENERAL_FILE} )
num_lines=$(( num_lines + 1 ))


# Perform docking
for (( i=1; i<= ${num_lines}; i++ ))
do
    # Make a copy of the template files
    cp ./input ./input.txt
    cp ./ligands.list ./ligands.list.txt
    
    line=$( tail -n+${i} ${GENERAL_FILE} | head -1 )
    
    # Replace the keywords in LeDock's INPUT file
    # Read the file line by line
    read var1 var2 var3 var4 <<< "$line"
    # var1: PDB ID
    # var2: COM x coordinate
    # var3: COM y coordinate
    # var4: COM z coordinate
    echo "Processing ${var1}"
    
    # Check if the element is present in the nohup.out file
    if grep -q "${var1}" "newfile.txt"; then
        # Element found in the file, so skip the current loop iteration
        continue
    fi
    
    MINX=$(awk "BEGIN {print $var2 - $spacing / 2}")
    MAXX=$(awk "BEGIN {print $var2 + $spacing / 2}")
    MINY=$(awk "BEGIN {print $var3 - $spacing / 2}")
    MAXY=$(awk "BEGIN {print $var3 + $spacing / 2}")
    MINZ=$(awk "BEGIN {print $var4 - $spacing / 2}")
    MAXZ=$(awk "BEGIN {print $var4 + $spacing / 2}")
    LIG_FILE="../../download/v2020-other-PL/$var1/${var1}_ligand.mol2"
    REP_FILE="../../download/v2020-other-PL/$var1/${var1}_pocket.pdb"
    
    # Run LePro to clean protein pdb file
    ./LePro ../../download/v2020-other-PL/${var1}/${var1}_protein.pdb

    # Process the variables as needed
    # Replacing keywords in the template files `input` and `ligands.list`
    sed -i "s/MIN_X/$MINX/g" input.txt
    sed -i "s/MAX_X/$MAXX/g" input.txt
    sed -i "s/MIN_Y/$MINY/g" input.txt
    sed -i "s/MAX_Y/$MAXY/g" input.txt
    sed -i "s/MIN_Z/$MINZ/g" input.txt
    sed -i "s/MAX_Z/$MAXZ/g" input.txt
    
    sed -i "s/rmsd/$rmsd/g" input.txt
    sed -i "s/NUM/$num/g" input.txt
    
    sed -i "s|receptor_file|pro.pdb|g" input.txt
    sed -i "s|ligand_file|${LIG_FILE}|g" ligands.list.txt

    # Run LeDock
    $LEDOCK input.txt
    OUTPUT_DOK="../../download/v2020-other-PL/${var1}/${var1}_ligand.dok"
    $LEDOCK -spli ${OUTPUT_DOK}
done

# Remove redundant intermediary files no longer used
rm ./dock.in
rm ./pro.pdb
rm ./input.txt
rm ./ligands.list.txt

echo " "
echo "Process exited successfully."
echo " "
