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
Preprocess the math500 dataset to parquet format, split by level
"""

import argparse
import os
from collections import defaultdict

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/math_level")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, split="test", trust_remote_code=True)
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def process_fn(example, idx):
        question = example.pop("problem")
        answer = example.pop("answer")
        solution = answer
        level = example.get("level", 1)  # Get level, default to 1 if not found
        data = {
            "data_source": "math",
            "ability": "math",
            "prompt": [
                {"role": "system", "content": instruction_following},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"split": "train", "index": idx, "level": level},
        }
        return data

    dataset = dataset.map(function=process_fn, with_indices=True)

    # Group dataset by level
    level_datasets = defaultdict(list)
    for example in dataset:
        level = example["extra_info"]["level"]
        level_datasets[level].append(example)
    
    print("Dataset distribution by level:")
    for level in sorted(level_datasets.keys()):
        print(f"Level {level}: {len(level_datasets[level])} examples")

    local_dir = args.local_dir
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Create datasets for each level
    for level in sorted(level_datasets.keys()):
        level_examples = level_datasets[level]
        level_dataset = datasets.Dataset.from_list(level_examples)
        
        # Create level subdirectory
        level_dir = os.path.join(local_dir, f"level_{level}")
        if not os.path.exists(level_dir):
            os.makedirs(level_dir)
        
        # Save both train.parquet and test.parquet (identical)
        train_path = os.path.join(level_dir, "train.parquet")
        test_path = os.path.join(level_dir, "test.parquet")
        
        level_dataset.to_parquet(train_path)
        level_dataset.to_parquet(test_path)
        
        print(f"Saved level {level} dataset: {len(level_dataset)} examples")
        print(f"  - {train_path}")
        print(f"  - {test_path}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)