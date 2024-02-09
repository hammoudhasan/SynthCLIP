# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Sample call:

python substr_matching.py --synthetic_captions_folder synthetic_captions \
                          --captions_with_count_folder synthetic_captions_with_count \
                          --metadata_filepath ./metadata.json \
                          --num_processes 8
                             

This code is used to generate captions using the Mistral-7B-Instruct-v0.2 model.
"""
import argparse
import json
import multiprocessing
import os
from itertools import repeat

from tqdm import tqdm

spaced_metadata = None


def spacing(text):
    puncts_to_wrap = [",", ".", ";", ":", "?", "!", "`"]
    chars_to_space = ["\t", "\n", "\r"]

    spaced_text = f" {text} "
    for punct_to_wrap in puncts_to_wrap:
        spaced_text = spaced_text.replace(punct_to_wrap, f" {punct_to_wrap} ")
    for char_to_space in chars_to_space:
        spaced_text = spaced_text.replace(char_to_space, " ")
    return spaced_text


def substr_matching(text, metadata):
    global spaced_metadata
    if spaced_metadata is None:
        spaced_metadata = []
        for entry in metadata:
            spaced_metadata.append(f" {entry} ")
    text = spacing(text)
    matched_entry_ids = []
    for entry_id, entry in enumerate(spaced_metadata):
        if entry in text:
            matched_entry_ids.append(entry_id)
    return matched_entry_ids


def dist_func(text, metadata):
    return [text, substr_matching(text, metadata)]


def main(args):

    synthetic_captions_folder = args.synthetic_captions_folder
    captions_with_count_folder = args.captions_with_count_folder
    metadata_filepath = args.metadata_filepath
    num_processes = int(args.num_processes)

    os.makedirs(captions_with_count_folder, exist_ok=True)

    with open(metadata_filepath, "r") as f:
        metadata = json.load(f)

    json_files = [
        f for f in os.listdir(synthetic_captions_folder) if f.endswith(".json")
    ]
    print(f"There are {len(json_files)} json files.")

    for file in json_files:
        with open(os.path.join(synthetic_captions_folder, file), "r") as f:
            parsed_json = json.load(f)

        raw_text = [
            text.replace('"', "").strip(" ").strip("\n")
            for text in parsed_json
            if "\n" not in text
        ]

        pool = multiprocessing.Pool(num_processes)
        text_with_count = pool.starmap(
            dist_func, tqdm(zip(raw_text, repeat(metadata)), total=len(raw_text))
        )

        with open(os.path.join(captions_with_count_folder, file), "w") as f:
            parsed_json = json.dump(text_with_count, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for substring matching.")
    parser.add_argument(
        "--synthetic_captions_folder",
        type=str,
        required=True,
        help="Name of the folder where the raw captions are",
    )
    parser.add_argument(
        "--captions_with_count_folder",
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
        "--num_processes",
        type=str,
        required=64,
        help="Path to metadata file",
    )

    args = parser.parse_args()

    main(args)
