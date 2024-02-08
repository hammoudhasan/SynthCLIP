# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import json
import os
import time
from contextlib import nullcontext
from itertools import islice

import cv2
import numpy as np
import torch
from einops import rearrange
from imwatermark import WatermarkEncoder
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from tqdm import tqdm, trange

torch.set_grad_enabled(False)

def get_rewrites_from_folder(rewrites_folder):
    json_files = glob.glob(os.path.join(rewrites_folder, f"*.json"))
    file_numbers = [int(file.split("/")[-1].split(".")[0]) for file in json_files]
    json_files = [file for _, file in sorted(zip(file_numbers, json_files))]

    text_list = []

    for file in json_files:
        with open(file, "r") as f:
            text_list.extend(json.load(f))

    text_list = [text.strip(" ").strip("\n").replace('"', "").replace('\t', ' ').replace('\r', ' ').replace('\n', ' ') for text in text_list]

    return text_list

def get_rewrites_from_file(captions_file):

    with open(captions_file, "r") as f:
        captions = json.load(f)

    captions = [text.strip(" ").strip("\n").replace('"', "").replace('\t', ' ').replace('\r', ' ').replace('\n', ' ') for text in captions]

    return captions

def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model

class StableGenerator(object):

    def __init__(self, model, opt):
        self.opt = opt
        # model
        self.model = model

        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        if opt.plms:
            sampler = PLMSSampler(model, device=device)
        elif opt.dpm:
            sampler = DPMSolverSampler(model, device=device)
        else:
            sampler = DDIMSampler(model, device=device)
        self.sampler = sampler

        # unconditional vector
        self.uc = model.get_learned_conditioning([""])
        if self.uc.ndim == 2:
            self.uc = self.uc.unsqueeze(0)
        self.batch_uc = None

        # shape
        self.shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        # precision scope
        self.precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext

    def generate(self, prompts, n_sample_per_prompt):
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():

                    # prepare the unconditional vector
                    bsz = len(prompts) * n_sample_per_prompt
                    if self.batch_uc is None or self.batch_uc.shape[0] != bsz:
                        self.batch_uc = self.uc.expand(bsz, -1, -1)

                    # prepare the conditional vector
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = self.model.get_learned_conditioning(prompts)
                    batch_c = c.unsqueeze(1).expand(-1, n_sample_per_prompt, -1, -1)
                    batch_c = batch_c.reshape(bsz, batch_c.shape[-2], batch_c.shape[-1])

                    # sampling
                    samples_ddim, _ = self.sampler.sample(S=self.opt.steps,
                                                          conditioning=batch_c,
                                                          batch_size=bsz,
                                                          shape=self.shape,
                                                          verbose=False,
                                                          unconditional_guidance_scale=self.opt.scale,
                                                          unconditional_conditioning=self.batch_uc,
                                                          eta=self.opt.ddim_eta,
                                                          x_T=None)     # no fixed start code

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    # x_samples_ddim = x_samples_ddim.cpu().numpy()
                    x_samples_ddim = 255. * x_samples_ddim

                    return x_samples_ddim

def batched_sd_generation(args, rewrites_list):

    config = OmegaConf.load(f"{args.config}")
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{args.ckpt}", device)

    # get the generator
    generator = StableGenerator(model, args)

    os.makedirs(os.path.join(args.save_path, str(args.chunk_idx)), exist_ok=True)

    for i in range(args.start_idx, args.end_idx, args.batch_size):

        rewrites_batch = rewrites_list[i : min(i + args.batch_size, args.end_idx)]
        
        start = time.time()

        images = generator.generate(rewrites_batch, n_sample_per_prompt=args.num_images_per_prompt)

        image_names = [
            f"{k}_{j}"
            for j in range(args.num_images_per_prompt)
            for k in range(i, min(i + args.batch_size, args.end_idx))
        ]

        for j in range(len(images)):
            x_sample = images[j]
            img = Image.fromarray(x_sample.astype(np.uint8))
            if args.save_resolution != args.H:
                img = img.resize((args.save_resolution, args.save_resolution))
            img.save(os.path.join(args.save_path, str(args.chunk_idx), f"{image_names[j]}.jpeg"))

        print(
            f"It took {time.time()-start} seconds to generate and save {len(rewrites_batch)} images."
        )

def main(opt):
    seed_everything(args.seed)
    # rewrites_list = get_prompts(args.captions_file)
    rewrites_list = get_rewrites_from_file(args.captions_file)

    assert len(rewrites_list) > 0

    start_idx = args.chunk_size * args.chunk_idx
    end_idx = min(start_idx + args.chunk_size, len(rewrites_list))
    assert start_idx + args.offset < end_idx
    start_idx = start_idx + args.offset

    args.start_idx = start_idx
    args.end_idx = end_idx
    args.batch_size = args.max_imgs_per_gpu // args.num_images_per_prompt

    batched_sd_generation(args, rewrites_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for SDgeneration.")

    # General args
    parser.add_argument(
        "--captions_file",
        required=True,
        type=str,
        help="Path to original captions file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Name of the folder where the images are going to be saved",
    )
    # Generation args
    parser.add_argument(
        "--scale",
        type=float,
        default=7.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--max_imgs_per_gpu",
        type=int,
        default=16,
        help="Max number of images that can be generated at selected resolution on a GPU.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--save_resolution", type=int, default=256, help="Saving resolution"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/ibex/ai/home/itanh0b/stablediffusion/SDv1.5/v1-5-pruned-emaonly.ckpt",
        help="path to the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    # Batching args
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of captions to rewrite at once.",
    )
    parser.add_argument(
        "--chunk_idx", type=int, default=0, help="Chunk index to rewrite."
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="How many images to skip."
    )
    args = parser.parse_args()

    main(args)

