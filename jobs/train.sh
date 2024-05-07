#!/bin/bash -l

#$ -N imae_mm
#$ -l h_rt=24:00:0
#$ -l mem=80G
#$ -l gpu=2
#$ -ac allow=EFL

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

export OMP_NUM_THREADS=2

# echo "This script is running on "
# hostname

echo "GPU information" 
nvidia-smi

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

# wandb login --relogin 8c3ad30d1b4df3c419d42a87c1979b0eb404232e

echo $(date +%d-%m-%Y_%H:%M:%S)
torchrun --nnodes=1 --nproc_per_node=2 ../program/imae/main.py\
        --train True\
        --epochs 600\
        --resume-epoch 0\
        --database moving_mnist
echo $(date +%d-%m-%Y_%H:%M:%S)