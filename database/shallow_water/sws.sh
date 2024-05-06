#!/bin/bash -l

#$ -N sws
#$ -l h_rt=1:00:00
#$ -l mem=100G
#$ -pe smp 4

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

python simulation.py --iteration-times 10000
status=$?
if [ $status -ne 0 ]; then
    echo "Simulation with 10000 iterations failed with status $status"
    exit $status
fi

python simulation.py --iteration-times 12000
status=$?
if [ $status -ne 0 ]; then
    echo "Simulation with 12000 iterations failed with status $status"
    exit $status
fi

python dataset_split.py

echo "All simulations completed successfully"