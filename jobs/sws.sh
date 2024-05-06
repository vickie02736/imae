#!/bin/bash -l

#$ -N shallow_water_simulation
#$ -l h_rt=1:00:00
#$ -l mem=100G
#$ -pe smp 2

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

# python ../database/shallow_water/simulation.py
python ../database/shallow_water/split_dataset.py

echo "All simulations completed successfully"