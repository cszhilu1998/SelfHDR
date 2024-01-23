#!/bin/bash
echo "Start to test the model...."

dataroot="/Data/dataset/HDR/Kalantari17"  # set the path of dataset
ckpt="./pretrained_models"

gpu="1"   # set the id of GPUs
iter="1"  # set the epoch number of the pre-trained model

name="ahdrnet_stage2"  # should be 'xxxx_stage2'
net="AHDRNet"  # can only choose 'AHDRNet', 'FSHDR', 'HDR-Transformer', or 'SCTNet'


python test.py \
    --dataset_name sig17align    --model selfhdr2       --name $name           --network $net      --dataroot $dataroot   \
    --checkpoints_dir $ckpt      --load_iter $iter      --save_imgs True       --gpu_ids $gpu  

