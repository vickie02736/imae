#!/bin/bash -l

#$ -N test_outer
#$ -l h_rt=24:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=L

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

echo "GPU information" 
nvidia-smi


source /home/uceckz0/miniconda3/bin/activate
conda activate imae

timestamp=$(date +%d-%m-%Y_%H:%M:%S)
echo $timestamp

for i in {1..9}; do
    mask_ratio=$(echo "scale=1; $i / 10" | bc)
    python ../program/test.py\
        --checkpoint-num 586\
        --rollout-times 5\
        --mask-ratio $mask_ratio\
        --task inner_rollout
done

echo $timestamps