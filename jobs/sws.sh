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

file_path="../dataset_split/txt/inner_test_file.txt"
mapfile -t pairs < "$file_path"

for pair in "${pairs[@]}"
do
    # Extract R and Hp values from the string
    R=$(echo "$pair" | sed 's/R_\([0-9]*\)_Hp_[0-9]*/\1/')
    Hp=$(echo "$pair" | sed 's/R_[0-9]*_Hp_\([0-9]*\)/\1/')

    echo "R = $R and Hp = $Hp $(date +%d-%m-%Y_%H:%M:%S)"
    python ../program/shallow_water_simulation.py\
        --R $R\
        --Hp $Hp\
        --root-path ../data/\
        --task-name shallow_water_simulation_inner_rollout\
        --dataset-name inner_rollout_test_file\
        --iteration-times 20000
done