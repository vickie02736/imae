#!/bin/bash -l

#$ -N imae_c
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
        --epochs 250\
        --mask-type consecutive
echo $timestamp