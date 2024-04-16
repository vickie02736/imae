#!/bin/bash -l

#$ -N imae
#$ -l h_rt=48:00:0
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
        --epochs 400\
        --start-epoch 200
echo $timestamp