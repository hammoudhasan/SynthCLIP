#!/bin/bash -l

## Run multinode distributed training 4 nodes 4 GPUs each with submitit for 1440 minutes = 24 hrs
python run_with_submitit.py \
    --dataset synthclip \
    --dataset-type csv \
    --train-data /ibex/ai/project/c2182/synthclip/datasets/CuratedSynthetic/csvs/curated-synthetic-captions-w2.csv \
    --output-dir vitb16-synthclip \
    --model CLIP_VITB16 \
    --epochs 40 \
    --warmup-epochs 1 \
    --batch-size 256 \
    --lr 5e-4 \
    --wd 0.5 \
    --workers 6 \
    --world-size 16 \
    --ngpus 4 \
    --nodes 4 \
    --timeout 1440 \
    --gpu_type v100 \
    --wandb

## Test on a single GPU
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --dataset synthclip \
#     --dataset-type csv \
#     --train-data /ibex/ai/project/c2182/synthclip/datasets/CuratedSynthetic/csvs/curated-synthetic-captions-w2.csv \
#     --output-dir vitb16-synthclip \
#     --model CLIP_VITB16 \
#     --epochs 40 \
#     --warmup-epochs 1 \
#     --batch-size 256 \
#     --lr 5e-4 \
#     --wd 0.5

## Test on a single node with 4 GPUs
# torchrun --nnodes=1 --nproc_per_node=4 main.py \
#          --dataset synthclip \
#          --dataset-type csv \
#          --train-data /ibex/ai/project/c2182/synthclip/datasets/CuratedSynthetic/csvs/curated-synthetic-captions-w2.csv \
#          --output-dir vitb16-synthclip \
#          --model CLIP_VITB16 \
#          --epochs 40 \
#          --warmup-epochs 1 \
#          --batch-size 256 \
#          --lr 5e-4 \
#          --wd 0.5