# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Sample call:

python balancing.py --captions_with_count_folder synthetic_captions_with_count \
                    --balanced_captions_folder balanced_curated_captions \
                    --metadata_filepath ./metadata.json \
                    --t 10
"""
import argparse
import json
import os
import random

import numpy as np
from tqdm import tqdm


def balance_sampling(matched_entry_ids, entry_prob):

    for entry_id in matched_entry_ids:
        if random.random() < entry_prob[entry_id]:
            return True

    return False

def main(args):
    
    captions_with_count_folder = args.captions_with_count_folder
    balanced_captions_folder = args.balanced_captions_folder
    metadata_filepath = args.metadata_filepath
    t = args.t

    os.makedirs(balanced_captions_folder, exist_ok=True)

    with open(metadata_filepath) as f:
        metadata = json.load(f)

    entry_count = np.zeros(shape=(len(metadata),), dtype=np.uint64)
    
    D = []
    captions = set()
    json_files = [f for f in os.listdir(captions_with_count_folder) if f.endswith(".json")]
    print(f"There are {len(json_files)} json files.")

    for file in json_files:
        with open(os.path.join(captions_with_count_folder, file)) as f:
            parsed_json = json.load(f)

        for rec in tqdm(parsed_json):
            if rec[0] in captions:
                continue
            else:
                captions.add(rec[0])
                for entry in rec[1]:
                    entry_count[entry] += 1
                D.append(rec)

    np.save(os.path.join(balanced_captions_folder, "entry_count.npy"), entry_count)

    with open(f"{balanced_captions_folder}/all_dedup_captions_with_count.json", "w") as f:
        json.dump(D, f)

    print(f"There are {len(D)} unique captions")

    entry_count[entry_count < t] = t
    entry_prob = t / entry_count

    D_star = []
    for rec in tqdm(D):
        if balance_sampling(rec[1], entry_prob):
            D_star.append(rec)

    print(f"Total of {len(D_star)} captions curated.")

    with open(os.path.join(balanced_captions_folder, f"curated_captions_with_count_{t}.json"), "w") as fw:
        json.dump(D_star, fw)
    
    curated_captions = [rec[0] for rec in D_star]
    with open(os.path.join(balanced_captions_folder, f"curated_captions_{t}.json"), "w") as fw:
        json.dump(curated_captions, fw)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Arguments for balancing captions."
    )
    parser.add_argument(
        "--captions_with_count_folder",
        type=str,
        required=True,
        help="Name of the folder where the raw captions are",
    )
    parser.add_argument(
        "--balanced_captions_folder",
        type=str,
        required=True,
        help="Name of the folder where the raw captions are",
    )
    parser.add_argument(
        "--metadata_filepath",
        type=str,
        required=True,
        help="Path to metadata file",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=10,
        help="Hyperparameter t in MetaCLIP; controls the probability of sampling captions with concepts.",
    )

    args = parser.parse_args()
    
    main(args)