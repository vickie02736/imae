#!/bin/bash -l

#$ -N imae_3
#$ -l h_rt=72:00:0
#$ -l mem=16G
#$ -l gpu=1
#$ -ac allow=EFL

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
python ../program/main.py\
        --mask-ratio 0.3\
        --epochs 200\
        --rollout-times 2\
        --batch-size 128
echo $timestamp