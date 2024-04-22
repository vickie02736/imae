#!/bin/bash -l

#$ -N sws
#$ -l h_rt=1:00:0
#$ -l mem=50G

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

declare -a pairs=('R_72_Hp_3' 'R_121_Hp_3'
                  'R_72_Hp_7' 'R_121_Hp_7'
                  'R_72_Hp_11' 'R_121_Hp_11'
                  'R_72_Hp_15' 'R_121_Hp_15')

for pair in "${pairs[@]}"
do
    # Extract R and Hp values from the string
    R=$(echo "$pair" | sed 's/R_\([0-9]*\)_Hp_[0-9]*/\1/')
    Hp=$(echo "$pair" | sed 's/R_[0-9]*_Hp_\([0-9]*\)/\1/')

    echo "Starting simulation with R = $R and Hp = $Hp at $(date +%d-%m-%Y_%H:%M:%S)"
    python ../program/shallow_water_simulation.py\
        --R $R\
        --Hp $Hp\
        --root-path ../data/\
        --task-name shallow_water_simulation_rollout_test
        --iteration-times 20000
    echo echo "Finished simulation with R = $R and Hp = $Hp at $(date +%d-%m-%Y_%H:%M:%S)"
done