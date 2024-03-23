#!/bin/bash -l

#$ -N imae_2
#$ -l h_rt=26:00:0

#$ -l gpu=1
#$ -ac allow=EF

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

python main_imae.py --mask_ratio 0.2 --epochs 200 --batch_size 64