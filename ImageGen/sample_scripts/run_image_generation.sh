#!/bin/bash -l
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 6
#SBATCH --constraint=[a100]

module load cuda/11.8

conda activate synthclip

python batched_txt2img.py --steps 50 \
                          --scale 2 \
                          --max_imgs_per_gpu 16 \
                          --balanced_captions_filepath ../TextGen/balanced_curated_captions/curated_captions.json \
                          --images_save_path synthetic_images \
                          --chunk_idx 0 \
                          --chunk_size 16 \