#!/bin/bash -l

#$ -N rollout_test
#$ -l h_rt=1:00:0
#$ -l mem=40G

#$ -l gpu=1
#$ -ac allow=L

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

python main_imae.py --mask_ratio 0.1 --epochs 10