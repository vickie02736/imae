#!/bin/bash -l

#$ -N convlstm
#$ -l h_rt=30:00:0
#$ -l mem=20G
#$ -l gpu=1
#$ -ac allow=EF
#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

# echo "This script is running on "
# hostname

# echo "GPU information" 
# nvidia-smi

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

echo $(date +%d-%m-%Y_%H:%M:%S)
torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py\
        --epochs 2\
        --resume-epoch 3\
        --database shallow_water\
        --save-frequency 2\
        --model-name cae_lstm\
        --interpolation linear
echo $(date +%d-%m-%Y_%H:%M:%S)

# model name: imae, convlstm, cae, cae_lstm 