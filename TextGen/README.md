# Caption Generation and Balancing

This README outlines the process for generating captions, substring matching, and balanced sampling of the generated synthetic captions.

### Step 1: Generate Captions

Start by generating captions using the `captions_generator.py` script. This will create a set of synthetic captions from the metadata.json file which is the concept bank.

```bash
python captions_generator.py --save_path synthetic_captions \
                             --generation_idx 0 \
                             --concept_bank_size -1 \
                             --metadata metadata.json
```

#### Parameters Explained

- `--save_path`: Directory where the generated captions will be saved.
- `--generation_idx`: Chunk index that will indicate the start index.
- `--concept_bank_size`: Number of captions to rewrite at once. (Recommended is -1) 
- `--metadata`: Path to the metadata JSON.

### Step 2: Concept Matching

Next, identify the concepts present in each caption using substring matching with the `substr_matching.py` script.

```bash
python substr_matching.py --synthetic_captions_folder synthetic_captions \
                          --captions_with_count_folder synthetic_captions_with_count \
                          --metadata_filepath ./metadata.json \
                          --num_processes 8
```

#### Parameters Explained

- `--synthetic_captions_folder`: Directory containing the generated captions.
- `--captions_with_count_folder`: Directory where the annotated captions will be saved.
- `--metadata_filepath`: Path to the metadata JSON file.
- `--num_processes`: Number of processes to use for parallel processing.

### Step 3: Balanced Sampling

Finally, perform balanced sampling based on the concept counts to ensure diversity in your captions.

```bash
python balancing.py --captions_with_count_folder synthetic_captions_with_count \
                    --balanced_captions_folder balanced_curated_captions \
                    --metadata_filepath ./metadata.json \
                    --t 10
```

#### Parameters Explained

- `--captions_with_count_folder`: Directory containing the annotated captions.
- `--balanced_captions_folder`: Directory where the balanced and curated captions will be saved.
- `--metadata_filepath`: Path to the metadata JSON file.
- `--t`: Threshold for balanced sampling criteria.

#### **Acknowledgement:** Parts of the image generation code were adopted from [MetaCLIP](https://github.com/facebookresearch/MetaCLIP).
---
