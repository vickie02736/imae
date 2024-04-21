#!/bin/bash -l

#$ -N test_9
#$ -l h_rt=00:5:0
#$ -l mem=8G
#$ -l gpu=1
#$ -ac allow=EFL

#$ -cwd

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

timestamp=$(date +%d-%m-%Y_%H:%M:%S)
echo $timestamp
python ../program/test.py\
    --checkpoint-num 470\
    --rollout-times 5\
    --mask-ratio 0.9
echo $timestamp