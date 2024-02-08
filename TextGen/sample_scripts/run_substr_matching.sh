#!/bin/bash -l
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --job-name=substr_matching
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8

conda activate synthclip

python substr_matching.py --synthetic_captions_folder synthetic_captions \
                          --captions_with_count_folder synthetic_captions_with_count \
                          --metadata_filepath ./metadata.json \
                          --num_processes 8