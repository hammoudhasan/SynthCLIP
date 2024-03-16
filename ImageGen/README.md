
# 🎨 Image Generation with `batched_txt2img.py`

Below is a guide on how to use the command, along with detailed explanations of each parameter for our image generation code. 

## **Command Usage**

Execute the following command in your terminal to image generation:

```bash
python batched_txt2img.py --steps 50 \
                          --scale 2 \
                          --max_imgs_per_gpu 16 \
                          --balanced_captions_filepath ../TextGen/balanced_curated_captions/curated_captions.json \
                          --images_save_path synthetic_images \
                          --chunk_idx 0 \
                          --chunk_size 1000 \
```

## 📖 **Parameters Explained**

- `--steps 50`: Diffusion steps.

- `--scale 2`: Diffusion guidance scale.

- `--max_imgs_per_gpu 16`: Num of images per gpu for batched generation.

- `--balanced_captions_filepath ../TextGen/balanced_curated_captions/curated_captions.json`: Path to captions.

- `--images_save_path`: Path to save images in.

- `--chunk_idx 0`: Chunk to process.

- `--chunk_size`: Number of images per chunk.


#### **Acknowledgement:** Parts of the image generation code were adopted from [StableRep](https://github.com/google-research/syn-rep-learn/tree/main/StableRep).
---
