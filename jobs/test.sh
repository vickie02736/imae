source /home/uceckz0/miniconda3/bin/activate
conda activate imae

echo $(date +%d-%m-%Y_%H:%M:%S)
torchrun --nnodes=1 --nproc_per_node=1 ../program/main.py\
         --test-flag True\
         --resume-epoch 1\
         --database shallow_water\
         --interpolation linear\
         --model-name convlstm

echo $(date +%d-%m-%Y_%H:%M:%S)

# model name: imae, convlstm, cae, cae_lstm 