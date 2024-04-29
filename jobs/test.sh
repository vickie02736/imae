#!/bin/bash -l

#$ -N inner_7
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

for i in {7..9}; do
    mask_ratio=$(echo "scale=1; $i / 10" | bc)
    python ../program/test.py\
        --checkpoint 400\
        --rollout-times 7\
        --mask-ratio $mask_ratio\
        --task inner\
        --mask-type random
done
# python ../program/test.py\
#     --checkpoint 400\
#     --rollout-times 7\
#     --mask-ratio 0.5\
#     --task inner_rollout\
#     --mask-type random
echo $timestamps