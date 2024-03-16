
# Training Code

This guide provides detailed instructions on how to train models using the SynthClip dataset with our provided script `run_with_submitit.py`.

## 🚀 Quick Start

To start training your model, execute the following command in your terminal:

```bash
python run_with_submitit.py \
    --dataset synthclip \
    --dataset-type csv \
    --train-data /path/to/dataset/curated-synthetic-captions-w2.csv \
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
```

## 📘 Parameters Explained

- `--dataset synthclip`: Specifies the dataset name.
- `--dataset-type csv`: Format of the dataset.
- `--train-data`: Path to the training data CSV file.
- `--output-dir`: Directory where the output will be saved.
- `--model`: Model architecture to use for training.
- `--epochs`: Number of training epochs.
- `--warmup-epochs`: Number of warm-up epochs.
- `--batch-size`: Batch size per iteration.
- `--lr`: Learning rate.
- `--wd`: Weight decay.
- `--workers`: Number of worker threads for data loading.
- `--world-size`: Total number of processes to run (sum of all processes across all nodes).
- `--ngpus`: Number of GPUs to use per node.
- `--nodes`: Number of nodes to use.
- `--timeout`: Timeout in minutes for the job.
- `--gpu_type`: Type of GPU to use.
- `--wandb`: Enables logging to Weights & Biases for experiment tracking.

#### **Acknowledgement:** Parts of the image generation code were adopted from [SLIP](https://github.com/facebookresearch/SLIP).
---
