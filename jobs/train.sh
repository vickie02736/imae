#!/bin/bash -l

#$ -N cae_sw
#$ -l h_rt=12:00:0
#$ -l mem=10G
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
        --resume-epoch 1\
        --database shallow_water\
        --save-frequency 1\
        --model-name cae_lstm\
        --interpolation linear
echo $(date +%d-%m-%Y_%H:%M:%S)

# model name: imae, convlstm, cae, cae_lstm 