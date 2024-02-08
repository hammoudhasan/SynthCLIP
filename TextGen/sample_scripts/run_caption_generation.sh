#!/bin/bash -l
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --job-name=caption_generation
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu 8
#SBATCH --constraint=[v100]

conda activate synthclip

python captions_generator.py --save_path synthetic_captions \
                             --generation_idx 0 \
                             --concept_bank_size 100 \
							 --metadata metadata.json
