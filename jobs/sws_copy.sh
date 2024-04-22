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



for R in 36 49 64 72 81 90 100 110 121 143 144 150 160 169 180 196
do
    for Hp in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
    do

            echo "Starting R = $R and Hp = $Hp at $(date +%d-%m-%Y_%H:%M:%S)"
            python ../program/shallow_water_simulation.py\
                --R $R\
                --Hp $Hp\
                --root-path ../data/\
                --task-name shallow_water_simulation\
                --iteration-times 10000
            echo echo "Finished R = $R and Hp = $Hp at $(date +%d-%m-%Y_%H:%M:%S)"
    done
done