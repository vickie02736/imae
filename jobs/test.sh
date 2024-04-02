#!/bin/bash -l

#$ -N imae_5_1
#$ -l h_rt=3:00:0
#$ -l mem=8G
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
python ../test.py --category 5 --mask_ratio 0.1 --rollout_times 2 --load_epoch 556
echo $timestamp
