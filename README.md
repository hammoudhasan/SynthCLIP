# SynthCLIP: Are We Ready For a Fully Synthetic CLIP Training?

![Alt text](./teaser.png)

[[Paper]](https://arxiv.org/abs/2402.01832)  

In this repository, we will share the data, code, and trained models for our work. Stay tuned and star!

## Abstract
We present SynthCLIP, a novel framework for training CLIP models with entirely synthetic text-image pairs, significantly departing from previous methods relying on real data. Leveraging recent text-to-image (TTI) generative networks and large language models (LLM), we are able to generate synthetic datasets of images and corresponding captions at any scale, with no human intervention. With training at scale, SynthCLIP achieves performance comparable to CLIP models trained on real datasets. We also introduce SynthCI-30M, a purely synthetic dataset comprising 30 million captioned images.

## Conda Environment Setup
```
conda create -n synthclip python=3.10 -y
conda activate synthclip

pip install https://github.com/vllm-project/vllm/releases/download/v0.3.0/vllm-0.3.0+cu118-cp310-cp310-manylinux1_x86_64.whl
pip uninstall torch -y
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip uninstall xformers -y
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Trained Models:
- ViT-B/16 Trained on SynthCI-10M can be found [here](https://drive.google.com/drive/folders/1sBnbczyDJUuGMKDOYeN0cOHYcq5L5xhr?usp=sharing).
- ViT-B/16 Trained on SynthCI-20M can be found [here](https://drive.google.com/drive/folders/1mXaooGAVJngm87xIjxPmzBnQV-019oET?usp=sharing).
- ViT-B/16 Trained on SynthCI-30M can be found [here](https://drive.google.com/drive/folders/1RP50tKvDPaiueYnfkh2gpHfUMAJAHwJo?usp=sharing).
- ViT-B/16 Trained on CC12M can be found [here](https://drive.google.com/drive/folders/1WwDWTAG6U9_CWhlPjChIJ6YHsQHrrLKF?usp=sharing).

## Citation

```
@misc{hammoud2024synthclip,
      title={SynthCLIP: Are We Ready for a Fully Synthetic CLIP Training?}, 
      author={Hasan Abed Al Kader Hammoud and Hani Itani and Fabio Pizzati and Philip Torr and Adel Bibi and Bernard Ghanem},
      year={2024},
      eprint={2402.01832},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
