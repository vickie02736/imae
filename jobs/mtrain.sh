#!/bin/bash -l

#$ -N imae
#$ -l h_rt=12:00:0
#$ -l mem=40G
#$ -l gpu=2
#$ -ac allow=EFL

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

export OMP_NUM_THREADS=2

# echo "This script is running on "
# hostname

# echo "GPU information" 
# nvidia-smi

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

# wandb login --relogin 8c3ad30d1b4df3c419d42a87c1979b0eb404232e

# timestamp=$(date +%d-%m-%Y_%H:%M:%S)
# echo $timestamp
torchrun --nnodes=1 --nproc_per_node=2 ../program/main.py\
        --train True\
        --epochs 2\
        --restart-epoch 8
# echo $timestamp