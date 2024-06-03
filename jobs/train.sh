#!/bin/bash -l

#$ -N imae_mm
#$ -l h_rt=1:00:0
#$ -l mem=80G
#$ -l gpu=1
#$ -ac allow=EFL


#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

# export OMP_NUM_THREADS=1

# echo "This script is running on "
# hostname

# echo "GPU information" 
# nvidia-smi

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

echo $(date +%d-%m-%Y_%H:%M:%S)
torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py\
        --epochs 2\
        --resume-epoch 0\
        --database shallow_water\
        --model-name imae
echo $(date +%d-%m-%Y_%H:%M:%S)