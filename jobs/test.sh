#!/bin/bash -l

#$ -N imae_5_1
#$ -l h_rt=3:00:0
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=L

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

timestamp=$(date +%d-%m-%Y_%H:%M:%S)
echo $timestamp
python ../program/test.py\
    --checkpoint-path /home/uceckz0/Project/imae/data/Vit_checkpoint/5/checkpoint_110.tar\
    --mask-ratio 0.5\
    --rollout-times 2
echo $timestamp