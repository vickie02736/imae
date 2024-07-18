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


convert_to_seconds() {
    date -d "${1//_/ }" +%s
}


start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Train start time: $start_time"
torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py\
        --epochs 600\
        --resume-epoch 1\
        --database shallow_water\
        --save-frequency 20\
        --model-name imae

# model name: imae, convlstm, cae, cae_lstm 

end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Train end time: $end_time"

start_seconds=$(convert_to_seconds "$start_time")
end_seconds=$(convert_to_seconds "$end_time")

difference_seconds=$((end_seconds - start_seconds))
hours=$((difference_seconds / 3600))
minutes=$(( (difference_seconds % 3600) / 60 ))
seconds=$((difference_seconds % 60))

echo "Train Time taken: $hours hours, $minutes minutes and $seconds seconds"


start_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Test start time: $start_time"

for ratio in $(seq 0.1 0.1 0.9); do
    torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py \
             --test-flag True \
             --resume-epoch 601 \
             --database shallow_water \
             --mask-ratio $ratio \
             --model-name imae
done

end_time=$(date +%Y-%m-%d_%H:%M:%S)
echo "Test end time: $end_time"

convert_to_seconds() {
    date -d "${1//_/ }" +%s
}


start_seconds=$(convert_to_seconds "$start_time")
end_seconds=$(convert_to_seconds "$end_time")

difference_seconds=$((end_seconds - start_seconds))
hours=$((difference_seconds / 3600))
minutes=$(( (difference_seconds % 3600) / 60 ))
seconds=$((difference_seconds % 60))

echo "Test Time taken: $hours hours, $minutes minutes and $seconds seconds"