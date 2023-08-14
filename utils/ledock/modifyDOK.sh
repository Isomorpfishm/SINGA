#!/bin/bash

# Create the 'new' directory if it doesn't exist
mkdir -p ../dataset/general

# Find all .pdb files in the 'v2020' directory and its subdirectories,
# whose filenames contain the string '_ligand_dock', and copy them to the 'new' directory
find ../../download/v2020-other-PL/ -type f -name "*_ligand_dock*.pdb" -exec cp {} ../dataset/general \;



# Specify the directory path
directory="../dataset/general"

# Loop through each file in the directory
for file in "$directory"/*; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then
        # Process the file as needed
        echo "$file"
        
        # Replace the string in the file
        sed -i 's/  CL\([^ ]*\) / CL\1  /g' "$file"
        
        # Add your custom logic here
    fi
done


