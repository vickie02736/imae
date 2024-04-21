#!/bin/bash -l

#$ -N sws
#$ -l h_rt=1:00:0
#$ -l mem=50G

#$ -m be
#$ -M uceckz0@ucl.ac.uk

#$ -cwd

echo "This script is running on "
hostname

source /home/uceckz0/miniconda3/bin/activate
conda activate imae

timestamp=$(date +%d-%m-%Y_%H:%M:%S)
echo $timestamp
python ../program/shallow_water_simulation.py
echo $timestamp