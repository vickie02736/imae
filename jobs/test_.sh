#!/bin/bash -l

#$ -N outer_7
#$ -l h_rt=3:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=EFJL

#$ -cwd

# echo "This script is running on "
# hostname

# echo "GPU information" 
# nvidia-smi


source /home/uceckz0/miniconda3/bin/activate
conda activate imae

timestamp=$(date +%d-%m-%Y_%H:%M:%S)
echo $timestamp

for i in {1..9}; do
    mask_ratio=$(echo "scale=1; $i / 10" | bc)
    python ../program/test.py\
        --mask-ratio $mask_ratio\
        --mask-type random\
        --checkpoint 400
done
echo $timestamps