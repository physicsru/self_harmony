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

import copy
import logging
import os
import re
import random
from collections import defaultdict, Counter
from typing import List, Optional, Union, Dict, Any

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format"""
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def create_adversarial_prompt(original_problem: str, original_answer: str, adversarial_instruction: str) -> str:
    """Create adversarial problem based on original problem and answer"""
    return (
        f"Based on this original problem: {original_problem}\n"
        f"And its answer: {original_answer}\n\n"
        f"{adversarial_instruction}\n"
        f"Create a mathematically equivalent variant that tests the same concepts "
        f"but uses different wording or numbers while maintaining the same answer type. "
        f"Output your variant problem clearly."
    )


def abstract_problem_rule_based(adversarial_output: str, abstraction_rules: List[str] = None) -> str:
    """Extract abstract version using <question>...</question> tags"""
    
    # Primary method: extract text between <question> and </question> tags
    question_pattern = r'<question>(.*?)</question>'
    matches = re.findall(question_pattern, adversarial_output, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Use the last question if multiple found
        extracted_question = matches[-1].strip()
        # Clean up whitespace and formatting
        extracted_question = ' '.join(extracted_question.split())
        return extracted_question
    
    # Fallback: try the old rule-based approach if no question tags found
    if abstraction_rules:
        for rule in abstraction_rules:
            matches = re.search(rule, adversarial_output, re.IGNORECASE)
            if matches:
                extracted = matches.group(1).strip()
                if not extracted.endswith('?') and not extracted.endswith('.'):
                    extracted += '.'
                return f"Find {extracted}"
    
    # Final fallback: extract question-like patterns
    sentences = adversarial_output.split('.')
    for sentence in sentences:
        sentence = sentence.strip()
        if any(word in sentence.lower() for word in ['find', 'calculate', 'determine', 'what', 'solve']):
            return sentence.strip() + '.'
    
    # Last resort: return first sentence
    return sentences[0].strip() + '.' if sentences else adversarial_output[:100] + '...'


def majority_vote_trio(answers: List[str]) -> tuple[str, int]:
    """Get majority vote from list of answers and return (answer, count)"""
    if not answers:
        return "", 0
    
    # Clean and normalize answers
    cleaned_answers = []
    for ans in answers:
        if ans is None or str(ans).strip() == '':
            continue
        boxed_ans = extract_boxed_answer(ans)
        if boxed_ans:
            cleaned_answers.append(boxed_ans.strip())
        else:
            # Fallback: use last line or full answer
            cleaned_answers.append(ans.strip().split('\n')[-1])
    
    if not cleaned_answers:
        return "", 0
    
    # Count occurrences
    counter = Counter(cleaned_answers)
    most_common = counter.most_common(1)[0]
    return most_common[0], most_common[1]


class TrioRLHFDataset(RLHFDataset):
    """
    Extended RLHFDataset that generates trio variants (X_ori, X_adv, X_abs) from original problems.
    This dataset processes each original problem through the trio pipeline and generates
    training samples with consistency-based pseudo-labels.
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        # Initialize trio-specific configuration
        self.trio_config = config.get("trio", {})
        self.num_rollouts = self.trio_config.get("num_rollouts", 8)
        self.adversarial_system_prompt = self.trio_config.get("adversarial_system_prompt", 
            "You are tasked with creating a mathematically equivalent variant of the given problem. "
            "The variant should test the same mathematical concepts but use different wording, "
            "numbers, or scenario while maintaining the same difficulty level and answer. "
            "Please wrap the abstract/core question in <question>...</question> tags for easy extraction. "
            "Let's think step by step and output the final answer within \\boxed{}.")
        self.abstraction_rules = self.trio_config.get("abstraction_rules", [
            r"Find\s+([^.]+)\.",
            r"What\s+is\s+([^?]+)\?",
            r"Calculate\s+([^.]+)\.",
            r"Determine\s+([^.]+)\.",
        ])
        self.pseudo_label_strategy = self.trio_config.get("pseudo_label_strategy", "random_source")
        
        # Initialize base dataset
        super().__init__(data_files, tokenizer, config, processor)
        
        # Generate trio data
        self._generate_trio_data()

    def _generate_trio_data(self):
        """Generate trio variants for all problems in the dataset"""
        logger.info(f"Generating trio data for {len(self.dataframe)} problems...")
        
        trio_samples = []
        
        for idx in range(len(self.dataframe)):
            if idx % 100 == 0:
                logger.info(f"Processing problem {idx}/{len(self.dataframe)}")
            
            original_data = self.dataframe[idx]
            original_prompt = original_data[self.prompt_key]
            
            # Extract the user's problem text
            user_problem = None
            for msg in original_prompt:
                if msg['role'] == 'user':
                    user_problem = msg['content']
                    break
            
            if user_problem is None:
                logger.warning(f"No user message found in prompt at index {idx}")
                continue
            
            # Process through trio pipeline
            trio_result = self._process_single_problem_trio(idx, original_prompt, user_problem)
            
            # Create training samples from trio result
            training_samples = self._create_training_samples_from_trio(trio_result, original_data)
            trio_samples.extend(training_samples)
        
        # Replace original dataframe with trio samples (8 per problem, not 24)
        logger.info(f"Generated {len(trio_samples)} original samples from {len(self.dataframe)} problems (8x rollouts, adv/abs created dynamically)")
        self.trio_samples = trio_samples
        self.original_dataframe = self.dataframe  # Keep reference to original
        
    def _process_single_problem_trio(self, problem_idx: int, original_prompt: List[Dict], user_problem: str) -> Dict[str, Any]:
        """Process a single problem through the trio pipeline (mock implementation)"""
        
        # Mock answer generation - in real implementation, this would call the model
        def mock_generate_answers(prompt_msgs: List[Dict], num_samples: int) -> List[str]:
            # Simulate different answers for the same problem
            base_answers = [
                "The answer is 42.",
                "After calculation, we get 42.",
                "Solving step by step: 42",
                "The final result is 42.",
                "Therefore, the answer is 42.",
                "We find that the solution is 42.",
                "The correct answer is 42.",
                "By solving, we obtain 42."
            ]
            return [f"{ans} (Mock answer {i+1})" for i, ans in enumerate(base_answers[:num_samples])]
        
        # Step 1: Generate answers for original problem (X_ori)
        answers_ori = mock_generate_answers(original_prompt, self.num_rollouts)
        
        # Step 2: Create adversarial variants for each answer
        answers_adv = []
        adversarial_problems = []
        
        for i, ans_ori in enumerate(answers_ori):
            # Create adversarial problem
            adversarial_instruction = create_adversarial_prompt(
                user_problem, ans_ori, self.adversarial_system_prompt
            )
            
            # Mock adversarial problem generation
            adversarial_problem = f"Variant of: {user_problem} (based on answer {i+1})"
            adversarial_problems.append(adversarial_problem)
            
            # Mock answer for adversarial problem
            ans_adv = f"Mock adversarial answer {i+1} for variant problem"
            answers_adv.append(ans_adv)
        
        # Step 3: Create abstracted versions (X_abs)
        answers_abs = []
        abstracted_problems = []
        
        for i, adv_problem in enumerate(adversarial_problems):
            # Extract abstract version using rule-based method
            abstract_problem = abstract_problem_rule_based(adv_problem, self.abstraction_rules)
            abstracted_problems.append(abstract_problem)
            
            # Mock answer for abstract problem
            ans_abs = f"Mock abstract answer {i+1} for abstract problem"
            answers_abs.append(ans_abs)
        
        # Step 4: Consistency checks
        maj_ori, count_ori = majority_vote_trio(answers_ori)
        maj_adv, count_adv = majority_vote_trio(answers_adv)
        maj_abs, count_abs = majority_vote_trio(answers_abs)
        
        # Rollout-wise consistency (within each source)
        ori_consistency = count_ori / len(answers_ori) if answers_ori else 0
        adv_consistency = count_adv / len(answers_adv) if answers_adv else 0
        abs_consistency = count_abs / len(answers_abs) if answers_abs else 0
        
        # Cross-source consistency
        cross_consistency = {
            'ori_adv': maj_ori == maj_adv,
            'ori_abs': maj_ori == maj_abs,
            'adv_abs': maj_adv == maj_abs,
            'all_three': maj_ori == maj_adv == maj_abs
        }
        
        consistency_data = {
            'majority_answers': {'ori': maj_ori, 'adv': maj_adv, 'abs': maj_abs},
            'majority_counts': {'ori': count_ori, 'adv': count_adv, 'abs': count_abs},
            'rollout_consistency': {'ori': ori_consistency, 'adv': adv_consistency, 'abs': abs_consistency},
            'cross_consistency': cross_consistency
        }
        
        return {
            'problem_index': problem_idx,
            'original_problem': user_problem,
            'original_prompt': original_prompt,
            'adversarial_problems': adversarial_problems,
            'abstracted_problems': abstracted_problems,
            # Remove pre-calculated answers and pseudo-labels
            # These will be generated dynamically during training
        }
    
    def _select_pseudo_label_source(self, consistency_data: Dict[str, Any]) -> str:
        """Select which source to use for pseudo-labeling"""
        
        if self.pseudo_label_strategy == "random_source":
            return random.choice(['ori', 'adv', 'abs'])
        elif self.pseudo_label_strategy == "best_consistency":
            consistencies = consistency_data['rollout_consistency']
            return max(consistencies.keys(), key=lambda k: consistencies[k])
        elif self.pseudo_label_strategy == "highest_count":
            counts = consistency_data['majority_counts']
            return max(counts.keys(), key=lambda k: counts[k])
        else:
            return 'ori'  # Default
    
    def _create_training_samples_from_trio(self, trio_result: Dict[str, Any], original_data: Dict) -> List[Dict]:
        """Create training samples - ONLY original problems, adv/abs created dynamically during training"""
        
        samples = []
        
        # Only create samples for original problem (X_ori) - 8 samples
        # The adv/abs problems will be created dynamically during training
        for i in range(self.num_rollouts):
            sample = copy.deepcopy(original_data)
            sample[self.prompt_key] = trio_result['original_prompt']  # Original prompt
            sample['trio_metadata'] = {
                'problem_index': trio_result['problem_index'],
                'source': 'ori',  # Only original source
                'rollout_index': i,
                'uid': f"{trio_result['problem_index']}_ori_{i}"
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.trio_samples)
    
    def __getitem__(self, idx):
        """Get item from trio samples"""
        sample = self.trio_samples[idx]
        
        # Process through parent's __getitem__ logic
        row_dict = copy.deepcopy(sample)
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # Add trio-specific metadata to the returned item
        trio_metadata = sample.get('trio_metadata', {})
        row_dict["index"] = trio_metadata.get('uid', idx)
        row_dict["trio_metadata"] = trio_metadata
        row_dict["tools_kwargs"] = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        
        # For compatibility with existing reward manager
        row_dict["uid"] = trio_metadata.get('uid', f"trio_{idx}")
        
        return row_dict

    def __getstate__(self):
        """Handle serialization for multiprocessing"""
        state = self.__dict__.copy()
        # Remove large objects that can be reconstructed
        if not self.serialize_dataset:
            if "dataframe" in state:
                del state["dataframe"]
            if "trio_samples" in state:
                del state["trio_samples"]
        return state

    def __setstate__(self, state):
        """Handle deserialization"""
        self.__dict__.update(state)
        if not hasattr(self, 'trio_samples'):
            self._read_files_and_tokenize()
            self._generate_trio_data()