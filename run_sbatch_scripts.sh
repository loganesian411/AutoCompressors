#!/bin/bash

# Set the directory containing your sbatch scripts
SCRIPT_DIR="./sbatch_scripts"

# Change to the script directory
cd "$SCRIPT_DIR" || { echo "Directory not found: $SCRIPT_DIR"; exit 1; }

# Loop through all sbatch scripts in the directory
for script in *.sbatch; do
    # Check if there are any sbatch scripts
    if [[ -e "$script" ]]; then
        echo "Submitting script: $script"
        sbatch "$script"
    else
        echo "No sbatch scripts found in the directory."
        exit 1
    fi
done

echo "All sbatch scripts have been submitted."
