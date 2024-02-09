"""
Minimal code for loading SynthCI
"""

from PIL import Image
import pandas as pd
import logging
import zipfile
import os
from torch.utils.data import Dataset
from huggingface_hub import snapshot_download
from huggingface_hub import hf_hub_download
from torchvision import transforms


class CsvDataset(Dataset):
    def __init__(
        self,
        input_filename,
        transforms,
        img_key,
        caption_key,
        prefix_path,
        sep="\t",
        tokenizer=None,
    ):
        logging.debug(f"Loading csv data from {input_filename}.")
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.prefix_path = prefix_path

        logging.debug("Done loading data.")

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(
            Image.open(self.prefix_path + str(self.images[idx])).convert("RGB")
        )
        # With tokenization
        # texts = self.tokenize([str(self.captions[idx])])[0]
        # Without tokenization
        texts = str(self.captions[idx])
        return images, texts


if __name__ == "__main__":
    # Download the dataset
    REPO_ID = "hammh0a/SynthCLIP"

    # Uncomment for full dataset download
    # snapshot_download(repo_id=REPO_ID, repo_type="dataset", cache_dir="./cache/", local_dir_use_symlinks=False, local_dir="./synthclip_data/")

    # Download only ./synthclip_data/data/0.zip and ./synthclip_data/combined_images_and_captions.csv
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir="./cache/",
        local_dir_use_symlinks=False,
        local_dir="./synthclip_data/",
        filename="./SynthCI-30/data/0.zip",
    )
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        cache_dir="./cache/",
        local_dir_use_symlinks=False,
        local_dir="./synthclip_data/",
        filename="./SynthCI-30/combined_images_and_captions.csv",
    )

    prefix = "./synthclip_data/SynthCI-30/data/"

    # Inside ./synthclip_data/data there will be multiple zip files unzip all
    # Unzip the files
    for file in os.listdir(prefix):
        if file.endswith(".zip"):
            with zipfile.ZipFile(prefix + file, "r") as zip_ref:
                zip_ref.extractall(prefix)

    # Remove the zip files
    for file in os.listdir(prefix):
        if file.endswith(".zip"):
            os.remove(prefix + file)

    # transforms
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Load the dataset
    dataset = CsvDataset(
        input_filename="./synthclip_data/SynthCI-30/combined_images_and_captions.csv",
        transforms=transform,
        img_key="image_path",
        caption_key="caption",
        prefix_path=prefix,
    )

    img, caption = dataset[0]

    # visualize the image
    import matplotlib.pyplot as plt

    plt.imshow(img.permute(1, 2, 0))
    plt.title(caption)
    plt.savefig("sample.png")
    plt.show()
