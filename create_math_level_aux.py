import pandas as pd
import copy
import os

def create_train_aux_parquet(input_path, output_path):
    """
    Create train_aux.parquet based on train.parquet with modified system prompt
    for dynamic rewrite generation.
    """
    # Load the original dataset
    df = pd.read_parquet(input_path)
    
    # Create a copy for modification
    df_aux = df.copy()
    
    # Define the new system prompt for dynamic rewrite generation
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

def create_all_math_level_aux():
    """Create train_aux.parquet for all math levels 1-5"""
    base_dir = "data/math_level"
    
    for level in range(1, 6):
        level_dir = os.path.join(base_dir, f"level_{level}")
        input_path = os.path.join(level_dir, "train.parquet")
        output_path = os.path.join(level_dir, "train_aux.parquet")
        
        if os.path.exists(input_path):
            print(f"\n=== Creating auxiliary dataset for Math Level {level} ===")
            df_aux = create_train_aux_parquet(input_path, output_path)
            print(f"Level {level}: {len(df_aux)} samples processed")
        else:
            print(f"Warning: {input_path} not found. Skipping Level {level}")
    
    print("\n=== All auxiliary datasets created successfully! ===")

if __name__ == "__main__":
    create_all_math_level_aux()