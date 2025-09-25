# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the MMLU-Pro dataset into a single directory with train/test parquet files.
This version correctly handles the list-based options format.
"""

import argparse
import os
from collections import Counter

import datasets

# This is a placeholder for your HDFS utility.
# You can comment out the HDFS-related lines if you don't need them.
try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    print("Warning: 'verl.utils.hdfs_io' not found. HDFS functionality will be disabled.")
    makedirs = None
    copy = None


def process_fn(example, idx):
    """
    Processes a single example from the dataset into the target format.
    """
    # Extract original fields from the example
    question_text = example.pop("question")
    # MODIFIED: Treat `options` as a list of strings directly.
    option_lines = example.pop("options")
    answer_letter = example.pop("answer")
    category = example.pop("category")

    # Add letter prefixes to each option string in the list
    formatted_options = []
    for i, line in enumerate(option_lines):
        letter = chr(ord('A') + i)
        # Ensure the line is a string and stripped of whitespace
        formatted_options.append(f"{letter}. {str(line).strip()}")
    
    # Join the newly formatted options back into a single string for the prompt
    options_text = "\n".join(formatted_options)
    
    # Combine the question and options for the user prompt
    full_prompt_text = f"{question_text}\n{options_text}"

    # Instruction to ask only for the letter in a box
    instruction_following = "Choose the single correct option. Your final answer should be only the letter of the correct option, enclosed in a \\boxed{} block."

    # ground_truth is just the answer letter
    ground_truth = answer_letter

    # Construct the final data structure
    data = {
        "data_source": "mmlu-pro",
        "ability": category,
        "prompt": [
            {"role": "system", "content": instruction_following},
            {"role": "user", "content": full_prompt_text},
        ],
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"split": "test", "index": idx, "category": category},
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/mmlu_pro", help="Local directory to save the processed parquet files.")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy the data to (optional).")

    args = parser.parse_args()

    data_source = "TIGER-Lab/MMLU-Pro"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # MODIFIED: Removed the deprecated `trust_remote_code` argument.
    dataset = datasets.load_dataset(data_source, split="test")
    
    # MODIFIED: Check and report the distribution of option counts
    print("\nValidating number of choices per question...", flush=True)
    option_counts = Counter()
    for example in dataset:
        # MODIFIED: Get the length directly from the list.
        num_options = len(example['options'])
        option_counts[num_options] += 1
    
    print("Distribution of option counts:")
    if len(option_counts) == 1:
        print(f"  All {len(dataset)} questions have {list(option_counts.keys())[0]} options. ✅")
    else:
        for count, num_questions in sorted(option_counts.items()):
            print(f"  - {num_questions} questions have {count} options.")
    
    print("\nProcessing dataset...", flush=True)
    processed_dataset = dataset.map(function=process_fn, with_indices=True, num_proc=os.cpu_count())

    local_dir = args.local_dir
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"Created local directory: {local_dir}")

    print(f"\nSaving all {len(processed_dataset)} examples to parquet files...", flush=True)
    
    train_path = os.path.join(local_dir, "train.parquet")
    test_path = os.path.join(local_dir, "test.parquet")
    
    processed_dataset.to_parquet(train_path)
    processed_dataset.to_parquet(test_path)
    
    print("Saved dataset successfully:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

    if args.hdfs_dir and makedirs and copy:
        print(f"\nCopying processed data from '{local_dir}' to HDFS at '{args.hdfs_dir}'...", flush=True)
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print("Copy to HDFS complete.")
    
    print("\nProcessing finished successfully. 🎉")