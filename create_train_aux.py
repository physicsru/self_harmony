"""
Training Data Auxiliary Generator

This module creates auxiliary training datasets by modifying system prompts to generate
reframed versions of problems. The core idea is to encourage models to approach problems
from different perspectives while maintaining mathematical correctness.

Key Features:
- Transforms original problem statements into creative, alternative formulations
- Preserves exact mathematical logic and final answers
- Uses "The Reframer" persona to guide creative problem transformation
- Generates training data for self-harmony learning approaches

Usage:
    python create_train_aux.py

The script will:
1. Load the original training dataset (GPQA Diamond format)
2. Apply the reframer system prompt to each sample
3. Save the modified dataset for training auxiliary models
"""

import pandas as pd
import copy

def create_train_aux_parquet(input_path, output_path):
    """
    Create train_aux.parquet based on train_original.parquet with modified system prompt
    for dynamic rewrite generation.
    """
    # Load the original dataset
    df = pd.read_parquet(input_path)
    
    # Create a copy for modification
    df_aux = df.copy()
    
    
    new_system_prompt = """You are The Reframer, an AI that solves problems with creative flair. Your challenge is not just to find the answer, but to find a more interesting way to get there.

    Your Mission:

    Transform: Don't solve the ORIGINAL PROBLEM directly. First, rewrite it using one of the Strategies below to unlock a new perspective.

    Solve: Solve the new problem you created. Your solution's style must match your transformation.

    The Strategies (Your Toolkit)

    Concretize: Turn an abstract idea into a tangible story.

    Generalize: Turn a specific case into a general formula or algorithm to be solved.

    Domain Shift: Change the field, for example, from math to code, or from logic to a game.

    Add Noise: Weave in red herrings to test the ability to focus.

    Reverse: Start from the end and work backward.

    Incremental Complexity: Reframe the task as a two-stage challenge within the narrative: first solve a simple "warm-up" version to find the pattern, then immediately apply that pattern to the original, full-scale numbers.

    Focus on Constraints: Frame the problem as a puzzle about navigating strict rules.

    POV Shift: Narrate from the perspective of an element inside the problem.

    The Rules:

    You must state your chosen Strategy first.

    The core logic, numbers, and final boxed answer must be consistent with the original problem.

    Don't copy 5+ consecutive words from the original.

    RESPONSE FORMAT:
    STRATEGY: [Name of the chosen strategy]

    REWRITTEN PROBLEM:
    [Your transformed problem statement.]

    SOLUTION:
    [Your step-by-step solution that matches the style of your new problem.]
    \\boxed{[The final answer.]}

    ORIGINAL PROBLEM:\n"""
    


    # Modify the system prompt for each row
    for idx in range(len(df_aux)):
        prompt_array = copy.deepcopy(df_aux.at[idx, 'prompt'])
        # Update the system message
        prompt_array[0]['content'] = new_system_prompt
        df_aux.at[idx, 'prompt'] = prompt_array
    
    # Save the modified dataset
    df_aux.to_parquet(output_path, index=False)
    print(f"Created {output_path} with {len(df_aux)} samples")
    print(f"Modified system prompt for dynamic rewrite generation")
    
    return df_aux

if __name__ == "__main__":
    input_path = "data/train_gpqa_diamond.parquet"
    output_path = "data/train_gpqa_diamond_aux.parquet"
    
    df_aux = create_train_aux_parquet(input_path, output_path)
    
    # Show a sample of the modified prompt
    print("\nSample of modified system prompt:")
    print(df_aux.iloc[0]['prompt'][0]['content'])
    print(df_aux.iloc[0]['prompt'])
