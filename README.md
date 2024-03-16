<div align="center">

# SynthCLIP: Are We Ready For a Fully Synthetic CLIP Training? 

<div>
  <a href="https://scholar.google.com/citations?user=Plf1JSIAAAAJ&hl=en">Hasan Abed Al Kader Hammoud</a><sup>1*</sup>&nbsp;&nbsp;
  <a href="https://cemse.kaust.edu.sa/ece/people/person/hani-itani">Hani Itani</a><sup>1*</sup>&nbsp;&nbsp;
  <a href="https://fabvio.github.io/">Fabio Pizzati</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://scholar.google.com/citations?user=kPxa2w0AAAAJ&hl=en">Philip Torr</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://www.adelbibi.com/">Adel Bibi</a><sup>2</sup>&nbsp;&nbsp;
  <a href="https://www.bernardghanem.com/">Bernard Ghanem</a><sup>1</sup>
  <br>
  <sup>1</sup> KAUST,
  <sup>2</sup> University of Oxford,
</div>

---

<img src="./teaser.png" alt="SynthCLIP Teaser" width="500"> <!-- Sets the width to 500 pixels -->

[![Paper](https://img.shields.io/badge/arXiv-Paper-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2402.01832) 
[![GitHub stars](https://img.shields.io/github/stars/hammoudhasan/SynthCLIP?style=for-the-badge)](https://github.com/hammoudhasan/SynthCLIP/stargazers)

🔥 **Stay tuned for updates, and don't forget to star this repo for the latest on SynthCLIP!** 🔥

</div>

## 📜 Abstract
We present SynthCLIP, a novel framework for training CLIP models with entirely synthetic text-image pairs, significantly departing from previous methods relying on real data. Leveraging recent text-to-image (TTI) generative networks and large language models (LLM), we are able to generate synthetic datasets of images and corresponding captions at any scale, with no human intervention. With training at scale, SynthCLIP achieves performance comparable to CLIP models trained on real datasets. We also introduce SynthCI-30M, a purely synthetic dataset comprising 30 million captioned images.


## 🚀 Getting Started

### Environment Setup
First, let's set up the Conda environment to get you up and running:

```bash
conda create -n synthclip python=3.10 -y
conda activate synthclip

pip install https://github.com/vllm-project/vllm/releases/download/v0.3.0/vllm-0.3.0+cu118-cp310-cp310-manylinux1_x86_64.whl
pip uninstall torch -y
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip uninstall xformers -y
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

### 📦 Trained Models
Jumpstart your experiments with our pre-trained models:

- **ViT-B/16** Trained on **SynthCI-10M** ➡️ [Download](https://drive.google.com/drive/folders/1sBnbczyDJUuGMKDOYeN0cOHYcq5L5xhr?usp=sharing)
- **ViT-B/16** Trained on **SynthCI-20M** ➡️ [Download](https://drive.google.com/drive/folders/1mXaooGAVJngm87xIjxPmzBnQV-019oET?usp=sharing)
- **ViT-B/16** Trained on **SynthCI-30M** ➡️ [Download](https://drive.google.com/drive/folders/1RP50tKvDPaiueYnfkh2gpHfUMAJAHwJo?usp=sharing)
- **ViT-B/16** Trained on **CC12M** ➡️ [Download](https://drive.google.com/drive/folders/1WwDWTAG6U9_CWhlPjChIJ6YHsQHrrLKF?usp=sharing)

## 📖 Citation
If you find SynthCLIP useful in your research, please consider citing:

```bibtex
@misc{hammoud2024synthclip,
      title={SynthCLIP: Are We Ready for a Fully Synthetic CLIP Training?}, 
      author={Hasan Abed Al Kader Hammoud and Hani Itani and Fabio Pizzati and Philip Torr and Adel Bibi and Bernard Ghanem},
      year={2024},
      eprint={2402.01832},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

---
