#!/bin/bash -l

#$ -N test_imae_1
#$ -l h_rt=3:00:0
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=L

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "GPU information" 
nvidia-smi

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

timestamp=$(date +%d-%m-%Y_%H:%M:%S)
echo $timestamp
python ../program/test.py\
    --checkpoint-num 154\
    --rollout-times 2\
    --mask-ratio 0.1\
echo $timestamp