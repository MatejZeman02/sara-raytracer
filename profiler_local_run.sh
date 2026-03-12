# !/bin/bash

echo "Note: This script is intended to be run on a local machine with NVIDIA GPU and Nsight Compute installed (conda/system)."

# run the profiler on the main.py script in src
ncu -f --profile-from-start off -o profile_report --set full conda run -n raytracer python ./src/main.py

# open nsight ui
ncu-ui profile_report.ncu-rep

# remove the report file after closing ui?
# read -p "Do you want to remove report after closing? (y/n) " answer
# if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
#     rm profile_report.ncu-rep
# fi
