#!/bin/bash -l
#SBATCH --output=slurm/%x.%3a.%A.out
#SBATCH --error=slurm/%x.%3a.%A.err
#SBATCH --job-name=balancing
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8

conda activate synthclip

python balancing.py --captions_with_count_folder synthetic_captions_with_count \
                    --balanced_captions_folder balanced_curated_captions \
                    --metadata_filepath ./metadata.json \
                    --t 10