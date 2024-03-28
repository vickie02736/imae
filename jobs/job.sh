#!/bin/bash -l

#$ -N imae_9
#$ -l h_rt=72:00:0
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
python ../main_imae.py --mask_ratio 0.9 --epochs 600
echo $timestamp
