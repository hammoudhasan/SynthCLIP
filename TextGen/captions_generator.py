"""
Sample call:

python captions_generator.py --save_path synthetic_captions \
                             --generation_idx 0 \
                             --concept_bank_size -1 \
                             --metadata metadata.json

This code is used to generate captions using the Mistral-7B-Instruct-v0.2 model.
"""

import argparse
import json
import os
import random

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

# check directory exists ./LLMs/mistral-7b-instruct-v0.2
if not os.path.isdir("./LLMs/Mistral-7B-Instruct-v0.2"):

    os.makedirs("./LLMs/Mistral-7B-Instruct-v0.2", exist_ok=True)

    snapshot_download(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        local_dir="./LLMs/Mistral-7B-Instruct-v0.2",
        local_dir_use_symlinks=False,
        cache_dir="./cache/",
    )


class Captions_Generator:

    def __init__(self, args):

        self.args = args
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            presence_penalty=1.0,
            frequency_penalty=1.0,
        )

        seed = random.randint(0, 1000000)

        self.llm = LLM(
            model="./LLMs/Mistral-7B-Instruct-v0.2",
            dtype="float16",
            seed=seed,
            tensor_parallel_size=1,
        )

    def create_system_message(self, concept):

        system_msg = f"""Your task is to write me an image caption that includes and visually describes a scene around a concept. Your concept is: {concept}. Output one single grammatically caption that is no longer than 15 words. Do not output any notes, word counts, facts, etc. Output one single sentence only."""
        return system_msg

    def generate_captions(self, concepts):

        save_path = os.path.join(self.args.save_path)
        os.makedirs(save_path, exist_ok=True)

        batches = []
        for concept in concepts:
            system_msg = self.create_system_message(concept)
            batches.append("<s> [INST] " + system_msg + " [/INST]")

        outputs = self.llm.generate(batches, self.sampling_params)
        response = [output.outputs[0].text for output in outputs]

        with open(
            os.path.join(save_path, f"{self.args.generation_idx}.json"), "w"
        ) as f:
            json.dump(response, f, indent=2)


def main(args):

    rewriter = Captions_Generator(args)

    with open(args.metadata_filepath, "r") as f:
        concepts = json.load(f)

    if args.concept_bank_size != -1:
        concepts = random.sample(concepts, args.concept_bank_size)

    rewriter.generate_captions(concepts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for synthetic caption generation using Open Source LLM and VLLM."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Name of the folder where the rewrites will be saved",
    )
    parser.add_argument(
        "--metadata_filepath",
        type=str,
        required=True,
        help="Path to metadata file",
    )
    parser.add_argument(
        "--generation_idx",
        type=int,
        default=0,
        help="Chunk index that will indicate the start index.",
    )
    parser.add_argument(
        "--concept_bank_size",
        type=int,
        default=-1,
        help="Number of captions to rewrite at once.",
    )

    args = parser.parse_args()

    main(args)
