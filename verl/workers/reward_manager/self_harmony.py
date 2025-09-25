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

from collections import defaultdict, Counter
import re

import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

# Math equivalence checking imports - aligned with TTRL approach
try:
    from verl.utils.reward_score.ttrl_math.math_utils import extract_boxed_answer, grade_answer_mathd, grade_answer_sympy, is_latex_equal
    TTRL_MATH_AVAILABLE = True
except ImportError:
    TTRL_MATH_AVAILABLE = False

try:
    # TTRL-style normalization for fast grouping operations
    from verl.utils.reward_score.ttrl_math import simplify_expression_string
    TTRL_SIMPLIFY_AVAILABLE = True
except ImportError:
    TTRL_SIMPLIFY_AVAILABLE = False

try:
    # TTRL-style moderate math equivalence for reward assignment
    from verl.utils.reward_score.math import strip_string
    TTRL_STRIP_AVAILABLE = True
except ImportError:
    TTRL_STRIP_AVAILABLE = False

# Fallback to original math_verify if ttrl_math not available
try:
    from math_verify import verify, parse, ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False

# Cache for expensive math comparisons
from functools import lru_cache


def compare_ignoring_whitespace(string1, string2):
    """Removes all whitespace from two strings and compares them."""
    if string1 is None or string2 is None:
        return False
    # Use re.sub to find all whitespace characters (\s+) and replace them with nothing ('')
    squeezed_string1 = re.sub(r'\s+', '', str(string1))
    squeezed_string2 = re.sub(r'\s+', '', str(string2))
    return squeezed_string1 == squeezed_string2


def compare_without_textbox(string1, string2):
    """Removes all textbox from two strings and compares them."""
    if string1 is None or string2 is None:
        return False
    
    s1 = str(string1)
    s2 = str(string2)
    
    # Remove textbox patterns
    s1 = s1.replace("\\text{(", "")
    s2 = s2.replace("\\text{(", "")
    s1 = s1.replace(")}", "")
    s2 = s2.replace(")}", "")
    s1 = s1.replace("\\text{", "")
    s2 = s2.replace("\\text{", "")
    s1 = s1.replace("}", "")
    s2 = s2.replace("}", "")
    
    return compare_ignoring_whitespace(s1, s2)


@lru_cache(maxsize=1024)
def _fast_numeric_compare(str1, str2):
    """Fast numeric comparison for common cases."""
    try:
        # Try to parse as floats
        val1 = float(str1.replace(',', '').strip())
        val2 = float(str2.replace(',', '').strip())
        return abs(val1 - val2) < 1e-10
    except (ValueError, TypeError):
        pass

    try:
        # Try to parse as fractions
        from fractions import Fraction
        val1 = Fraction(str1.strip())
        val2 = Fraction(str2.strip())
        return val1 == val2
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    return False


@lru_cache(maxsize=512)
def _looks_like_math(text):
    """Fast heuristic to determine if text contains mathematical content."""
    if not text or len(text) < 2:
        return False

    # Check for mathematical indicators
    math_indicators = [
        # Numbers
        any(c.isdigit() for c in text),
        # Mathematical operators
        any(op in text for op in ['+', '-', '*', '/', '=', '^', '_']),
        # LaTeX/mathematical notation
        '\\' in text,
        # Mathematical brackets
        any(bracket in text for bracket in ['(', ')', '{', '}', '[', ']']),
        # Common mathematical words (only if combined with numbers)
        (any(word in text.lower() for word in ['frac', 'sqrt', 'log', 'sin', 'cos', 'tan']) and
         any(c.isdigit() for c in text))
    ]

    # Must have at least one mathematical indicator
    return any(math_indicators)


@lru_cache(maxsize=1024)
def _ttrl_fast_equiv(str1, str2):
    """TTRL-style fast equivalence for grouping operations.

    Uses simple normalization + string equality (like TTRL majority voting).
    Much faster than full math equivalence but less accurate.
    """
    if str1 is None or str2 is None:
        return False

    # Basic preprocessing
    s1 = str(str1).strip()
    s2 = str(str2).strip()

    if s1 == s2:
        return True

    # Try TTRL simplify_expression_string if available
    if TTRL_SIMPLIFY_AVAILABLE:
        try:
            s1_simplified = simplify_expression_string(s1)
            s2_simplified = simplify_expression_string(s2)
            return s1_simplified == s2_simplified
        except Exception:
            pass

    # Fallback to basic numeric comparison
    return _fast_numeric_compare(s1, s2)


@lru_cache(maxsize=1024)
def _ttrl_moderate_equiv(str1, str2):
    """TTRL-style moderate equivalence for reward assignment.

    Uses strip_string normalization + string equality (like TTRL is_equiv).
    More accurate than fast equiv but faster than full symbolic math.
    """
    if str1 is None or str2 is None:
        return False

    # Basic preprocessing
    s1 = str(str1).strip()
    s2 = str(str2).strip()

    if s1 == s2:
        return True

    # Try TTRL strip_string normalization if available
    if TTRL_STRIP_AVAILABLE:
        try:
            s1_normalized = strip_string(s1)
            s2_normalized = strip_string(s2)
            if s1_normalized == s2_normalized:
                return True
        except Exception:
            pass

    # Fallback to our existing fast comparisons
    if compare_ignoring_whitespace(s1, s2):
        return True
    if _fast_numeric_compare(s1, s2):
        return True
    if compare_without_textbox(s1, s2):
        return True

    # Additional mathematical equivalence for moderate mode
    # Try basic mathematical evaluation without expensive SymPy operations
    # Apply if either string looks mathematical OR if both could be numeric expressions
    should_try_math = (
        _looks_like_math(s1) or _looks_like_math(s2) or
        # Try for potential numeric comparisons (e.g., "4" vs "2+2")
        any(c.isdigit() or c in '+-*/^.' for c in s1 + s2)
    )

    if TTRL_SIMPLIFY_AVAILABLE and should_try_math:
        try:
            # Use simplify_expression_string which has timeout protection
            s1_simplified = simplify_expression_string(s1)
            s2_simplified = simplify_expression_string(s2)
            if s1_simplified == s2_simplified:
                return True
        except Exception:
            pass

    return False




def math_equivalent(answer1, answer2, mode='full'):
    """Check if two mathematical answers are equivalent using TTRL's approach.

    TTRL uses different equivalence levels for different stages:
    - 'fast': String-based equivalence for grouping operations (fast)
    - 'moderate': Normalization-based equivalence for reward assignment (balanced)
    - 'full': Full symbolic math equivalence for final evaluation (accurate)

    Args:
        answer1: First answer string
        answer2: Second answer string
        mode: Equivalence mode ('fast', 'moderate', or 'full')

    Returns:
        bool: True if answers are mathematically equivalent
    """
    if mode == 'fast':
        return _ttrl_fast_equiv(answer1, answer2)
    elif mode == 'moderate':
        return _ttrl_moderate_equiv(answer1, answer2)
    else:
        return _math_equivalent_full(answer1, answer2)


def _math_equivalent_full(answer1, answer2):
    """Full symbolic math equivalence checking (original implementation).

    Uses the same approach as validation ttrl_math.compute_score():
    1. Fast pre-checks for optimization
    2. Primary: ttrl_math grade_answer_mathd + grade_answer_sympy
    3. Handle boxed answers with extract_boxed_answer
    4. Use is_latex_equal for additional accuracy
    5. Fallback to math_verify if ttrl_math unavailable

    Args:
        answer1: First answer string
        answer2: Second answer string

    Returns:
        bool: True if answers are mathematically equivalent
    """
    if answer1 is None or answer2 is None:
        return False

    str1 = str(answer1).strip()
    str2 = str(answer2).strip()

    # Check if both are empty
    if str1 == '' and str2 == '':
        return True

    # Check if either is empty (but not both)
    if str1 == '' or str2 == '':
        return False

    # Direct string equality
    if str1 == str2:
        return True

    # Fast early exits (most common cases) - ordered by speed and likelihood
    # 2. Fast whitespace comparison (very common)
    if compare_ignoring_whitespace(str1, str2):
        return True

    # 3. Fast numeric comparison (handles most math answers)
    if _fast_numeric_compare(str1, str2):
        return True

    # 4. Fast textbox comparison
    if compare_without_textbox(str1, str2):
        return True

    # 5. Primary method: ttrl_math approach (aligned with validation) - only for math-like content
    if TTRL_MATH_AVAILABLE and (_looks_like_math(str1) or _looks_like_math(str2)):
        try:
            # Handle boxed answers first (most common case)
            processed_str1 = str1
            processed_str2 = str2
            if "\\boxed" in str2:
                processed_str2 = extract_boxed_answer(str2) or str2
            if "\\boxed" in str1:
                processed_str1 = extract_boxed_answer(str1) or str1

            # Try faster sympy first, then mathd only if needed
            if (grade_answer_sympy(processed_str1, processed_str2) or
                grade_answer_sympy(processed_str2, processed_str1)):
                return True

            # Only try expensive mathd if sympy failed and strings look complex
            if len(processed_str1) > 3 and len(processed_str2) > 3:
                if (grade_answer_mathd(processed_str1, processed_str2) or
                    grade_answer_mathd(processed_str2, processed_str1)):
                    return True

            # For additional accuracy, use LaTeX equivalence (non-fast mode) - only if processed differently
            if (processed_str1 != str1 or processed_str2 != str2) and ('\\' in processed_str1 or '\\' in processed_str2):
                if (is_latex_equal(processed_str1, processed_str2) or
                    is_latex_equal(processed_str2, processed_str1)):
                    return True
        except Exception:
            # If ttrl_math fails, continue with fallback methods
            pass

    # 6. Fallback to math_verify approach (slowest)
    if MATH_VERIFY_AVAILABLE:
        try:
            extraction_target = (ExprExtractionConfig(), LatexExtractionConfig(), StringExtractionConfig())
            if (verify(str1, str2, extraction_target) or
                verify(parse(str1), parse(str2), extraction_target)):
                return True
        except Exception:
            pass

    return False


def group_math_equivalent_answers(ans_list, choice_type="math", choice_size=4):
    """Optimized grouping using TTRL's fast equivalence approach.

    Uses fast string-based equivalence for grouping operations to minimize
    O(n²) complexity. This aligns with TTRL's approach for pseudo-labeling.

    Args:
        ans_list: List of answer strings
        choice_type: Type of comparison ("math" or "choice")
        choice_size: Number of choices for choice_type="choice" (4 for A-D, 8 for A-H, etc.)

    Returns:
        dict: Mapping from representative answer to list of equivalent answers
    """
    if not ans_list:
        return {}

    # Pre-filter and create index mapping for valid answers
    valid_answers = []
    original_indices = []

    for i, ans in enumerate(ans_list):
        if ans is not None and str(ans).strip() != '':
            valid_answers.append(str(ans).strip())
            original_indices.append(i)

    if not valid_answers:
        return {}

    groups = {}
    processed_indices = set()

    # Create equivalence classes using optimized approach
    for i, ans in enumerate(valid_answers):
        if i in processed_indices:
            continue

        # Start new equivalence class
        class_members = [ans]
        processed_indices.add(i)

        # Directly compare remaining candidates (skip pre-filtering for now)
        for j in range(i + 1, len(valid_answers)):
            if j in processed_indices:  # May have been processed by another class
                continue

            other_ans = valid_answers[j]
            # Use fast equivalence for grouping operations (TTRL approach)
            if math_equivalent_with_choice(ans, other_ans, choice_type, choice_size, mode='fast'):
                class_members.append(other_ans)
                processed_indices.add(j)

        # Use first answer as representative for this equivalence class
        groups[ans] = class_members

    return groups


@lru_cache(maxsize=2048)
def _should_skip_comparison(ans1, ans2):
    """Conservative heuristic to skip obviously different answers.

    Returns True only if answers are definitely different.
    Returns False if answers might be equivalent (err on the safe side).
    """
    # Only skip for very obvious cases to avoid false negatives

    # Skip if one is much longer AND they share no digits AND no mathematical symbols
    if abs(len(ans1) - len(ans2)) > max(len(ans1), len(ans2)) * 1.5:
        digits1 = set(c for c in ans1 if c.isdigit())
        digits2 = set(c for c in ans2 if c.isdigit())
        math_chars1 = set(c for c in ans1 if c in '+-*/=()[]{}.,\\')
        math_chars2 = set(c for c in ans2 if c in '+-*/=()[]{}.,\\')

        if (len(digits1 & digits2) == 0 and
            len(math_chars1 & math_chars2) == 0 and
            len(digits1) > 0 and len(digits2) > 0):
            return True

    # Skip if they're clearly text and completely different
    if (len(ans1) > 5 and len(ans2) > 5 and
        not any(c.isdigit() for c in ans1) and
        not any(c.isdigit() for c in ans2) and
        not any(c in '+-*/=()[]{}.,\\' for c in ans1) and
        not any(c in '+-*/=()[]{}.,\\' for c in ans2)):

        # Both are text-only, check if completely different
        words1 = set(ans1.lower().replace(' ', '').replace('-', '').replace('_', ''))
        words2 = set(ans2.lower().replace(' ', '').replace('-', '').replace('_', ''))

        if len(words1 & words2) / max(len(words1), len(words2), 1) < 0.3:
            return True

    # Conservative approach - don't skip in most cases
    return False


def majority_vote(ans_list, empty_value=''):
    ans_list = [a for a in ans_list if a is not None and str(a).strip() != '']
    if not ans_list:
        return empty_value
        
    # Group mathematically equivalent answers
    groups = group_math_equivalent_answers(ans_list)
    if not groups:
        return empty_value
    
    # Find the group with the most answers
    max_count = 0
    best_answer = empty_value
    
    for representative, group in groups.items():
        if len(group) > max_count:
            max_count = len(group)
            best_answer = representative
    
    return best_answer

def majority_vote_with_count(ans_list, empty_value='', choice_type="math", choice_size=4):
    """Majority vote that returns both answer and count (using math equivalence)"""
    ans_list = [a for a in ans_list if a is not None and str(a).strip() != '']
    if not ans_list:
        return empty_value, 0
        
    # Group mathematically equivalent answers
    groups = group_math_equivalent_answers(ans_list, choice_type, choice_size)
    if not groups:
        return empty_value, 0
    
    # Find the group with the most answers
    max_count = 0
    best_answer = empty_value
    
    for representative, group in groups.items():
        if len(group) > max_count:
            max_count = len(group)
            best_answer = representative
    
    return best_answer, max_count


def extract_choice_letter(answer_str, choice_size=4):
    """Extract choice letter from answer string for multiple choice questions.
    
    Args:
        answer_str: Answer string that may contain choice letters
        choice_size: Number of choices (4 for A-D, 8 for A-H, etc.)
        
    Returns:
        str: Normalized choice letter (A, B, C, ...) or original string if no choice found
    """
    if not answer_str or not isinstance(answer_str, str):
        return answer_str
    
    # Generate valid choices based on size
    valid_choices = [chr(ord('A') + i) for i in range(choice_size)]
    
    # Clean the answer string
    answer_clean = str(answer_str).strip().upper()
    
    # Look for choice patterns
    import re
    
    # Pattern 1: Single letter at start/end (A, B, C, etc.)
    single_letter_match = re.search(r'\b([A-Z])\b', answer_clean)
    if single_letter_match:
        letter = single_letter_match.group(1)
        if letter in valid_choices:
            return letter
    
    # Pattern 2: Choice with parentheses or brackets (A), [B], etc.
    paren_match = re.search(r'[\(\[\{]([A-Z])[\)\]\}]', answer_clean)
    if paren_match:
        letter = paren_match.group(1)
        if letter in valid_choices:
            return letter
    
    # Pattern 3: "The answer is A" or similar
    answer_is_match = re.search(r'(?:answer is|choice is|select|option)\s*([A-Z])', answer_clean, re.IGNORECASE)
    if answer_is_match:
        letter = answer_is_match.group(1)
        if letter in valid_choices:
            return letter
    
    # Pattern 4: Just the letter with period "A."
    letter_period_match = re.search(r'\b([A-Z])\.', answer_clean)
    if letter_period_match:
        letter = letter_period_match.group(1)
        if letter in valid_choices:
            return letter
    
    # If no choice pattern found, return original
    return answer_str


def choice_equivalent(answer1, answer2, choice_size=4):
    """Check if two answers are equivalent for multiple choice questions.
    
    Args:
        answer1: First answer string
        answer2: Second answer string  
        choice_size: Number of choices (4 for A-D, 8 for A-H, etc.)
        
    Returns:
        bool: True if answers represent the same choice
    """
    if answer1 is None or answer2 is None:
        return False
    
    choice1 = extract_choice_letter(answer1, choice_size)
    choice2 = extract_choice_letter(answer2, choice_size)
    
    # If both extracted valid choices, compare them
    valid_choices = [chr(ord('A') + i) for i in range(choice_size)]
    if choice1 in valid_choices and choice2 in valid_choices:
        return choice1 == choice2
    
    # Fall back to original string comparison if no valid choices extracted
    return str(answer1).strip().lower() == str(answer2).strip().lower()


def math_equivalent_with_choice(answer1, answer2, choice_type="math", choice_size=4, mode='full'):
    """Check if two answers are equivalent using specified comparison method.

    Args:
        answer1: First answer string
        answer2: Second answer string
        choice_type: Type of comparison ("math" or "choice")
        choice_size: Number of choices for choice_type="choice" (4 for A-D, 8 for A-H, etc.)
        mode: Equivalence mode for math comparisons ('fast', 'moderate', or 'full')

    Returns:
        bool: True if answers are considered equivalent
    """
    if choice_type == "choice":
        return choice_equivalent(answer1, answer2, choice_size)
    else:
        # Use specified math equivalence mode
        return math_equivalent(answer1, answer2, mode=mode)


def count_math_equivalent(ans_list, target_answer):
    """Count how many answers in the list are mathematically equivalent to target_answer.
    
    Args:
        ans_list: List of answer strings
        target_answer: The target answer to match against
        
    Returns:
        int: Number of mathematically equivalent answers
    """
    if not ans_list or target_answer is None:
        return 0
        
    count = 0
    for ans in ans_list:
        if math_equivalent(ans, target_answer):
            count += 1
    
    return count


def mutual_info_score(ori_ans_list, aug_ans_list, empty_value=''):
    """Select answer with highest mutual information F-score between two sources.
    
    For each answer a:
    score(a) = 4 * n0(a) * n1(a) / (n0(a) + n1(a))² + 1e-3 * (n0(a) + n1(a))
    
    This is monotonic with exact mutual information when N0 = N1 but much cheaper to compute.
    Frequency is incorporated as a small weighted component instead of tie-breaking.
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    if not ori_ans_list and not aug_ans_list:
        return empty_value
    
    # Count answers in each source
    ori_counts = Counter(ori_ans_list)
    aug_counts = Counter(aug_ans_list)
    
    # Get all unique answers
    all_answers = set(ori_counts.keys()) | set(aug_counts.keys())
    
    best_answer = empty_value
    best_score = -1
    best_frequency = 0
    
    for answer in all_answers:
        n0 = ori_counts.get(answer, 0)  # count in ori
        n1 = aug_counts.get(answer, 0)  # count in aug
        total_freq = n0 + n1
        
        if total_freq == 0:
            score = 0
        else:
            # F-score formula: 4 * n0 * n1 / (n0 + n1)²
            score = (4 * n0 * n1) / (total_freq * total_freq)
        
        # Combine score with small weighted frequency (1e-3 * total_freq)
        weighted_score = score + 1e-3 * total_freq
        
        # Update best answer (higher weighted score wins)
        if weighted_score > best_score:
            best_answer = answer
            best_score = weighted_score
            best_frequency = total_freq
    
    return best_answer


def compute_true_mutual_information(ori_ans_list, aug_ans_list, candidate_answer):
    """Compute true mutual information I(S; Ya) for a specific candidate answer.
    
    S = source indicator (0 for ori, 1 for aug)
    Ya = binary indicator (1 if answer equals candidate_answer, 0 otherwise)
    
    Args:
        ori_ans_list: List of answers from original branch (S=0)
        aug_ans_list: List of answers from augmented branch (S=1) 
        candidate_answer: The candidate answer to compute MI for
        
    Returns:
        Mutual information I(S; Ya) in nats
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    if not ori_ans_list or not aug_ans_list:
        return 0.0
        
    N0 = len(ori_ans_list)  # Total ori samples
    N1 = len(aug_ans_list)  # Total aug samples
    N_total = N0 + N1
    
    # Count occurrences of candidate_answer in each source
    n0_a = sum(1 for ans in ori_ans_list if ans == candidate_answer)  # n0(a)
    n1_a = sum(1 for ans in aug_ans_list if ans == candidate_answer)  # n1(a)
    
    # Build the 2x2 contingency table:
    # S=0     S=1
    # Ya=1   n0_a    n1_a
    # Ya=0   N0-n0_a N1-n1_a
    
    # Joint probabilities P(S, Ya)
    p_s0_ya1 = n0_a / N_total          # P(S=0, Ya=1)
    p_s1_ya1 = n1_a / N_total          # P(S=1, Ya=1)  
    p_s0_ya0 = (N0 - n0_a) / N_total   # P(S=0, Ya=0)
    p_s1_ya0 = (N1 - n1_a) / N_total   # P(S=1, Ya=0)
    
    # Marginal probabilities
    p_s0 = N0 / N_total                # P(S=0)
    p_s1 = N1 / N_total                # P(S=1)
    p_ya1 = (n0_a + n1_a) / N_total    # P(Ya=1)
    p_ya0 = 1.0 - p_ya1               # P(Ya=0)
    
    # Mutual information: I(S; Ya) = sum over all (s,y) of P(S=s, Ya=y) * log(P(S=s, Ya=y) / (P(S=s) * P(Ya=y)))
    mi = 0.0
    
    # Add epsilon to avoid log(0)
    epsilon = 1e-12
    
    # Term: P(S=0, Ya=1) * log(P(S=0, Ya=1) / (P(S=0) * P(Ya=1)))
    if p_s0_ya1 > epsilon and p_s0 > epsilon and p_ya1 > epsilon:
        mi += p_s0_ya1 * np.log(p_s0_ya1 / (p_s0 * p_ya1))
        
    # Term: P(S=1, Ya=1) * log(P(S=1, Ya=1) / (P(S=1) * P(Ya=1)))  
    if p_s1_ya1 > epsilon and p_s1 > epsilon and p_ya1 > epsilon:
        mi += p_s1_ya1 * np.log(p_s1_ya1 / (p_s1 * p_ya1))
        
    # Term: P(S=0, Ya=0) * log(P(S=0, Ya=0) / (P(S=0) * P(Ya=0)))
    if p_s0_ya0 > epsilon and p_s0 > epsilon and p_ya0 > epsilon:
        mi += p_s0_ya0 * np.log(p_s0_ya0 / (p_s0 * p_ya0))
        
    # Term: P(S=1, Ya=0) * log(P(S=1, Ya=0) / (P(S=1) * P(Ya=0)))
    if p_s1_ya0 > epsilon and p_s1 > epsilon and p_ya0 > epsilon:
        mi += p_s1_ya0 * np.log(p_s1_ya0 / (p_s1 * p_ya0))
    
    return max(0.0, mi)  # MI should be non-negative


def true_mutual_info_score(ori_ans_list, aug_ans_list, empty_value=''):
    """Select answer with LOWEST true mutual information with source indicator.
    
    We want the most "source-agnostic" answer - one that appears similarly 
    in both ori and aug branches. Lower MI = more balanced across sources.
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch  
        empty_value: Return value if no valid answer found
        
    Returns:
        Answer with lowest I(S; Ya), ties broken by total frequency
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    if not ori_ans_list and not aug_ans_list:
        return empty_value
    
    # Get all unique candidate answers
    all_candidates = set(ori_ans_list + aug_ans_list)
    
    if not all_candidates:
        return empty_value
        
    best_answer = empty_value
    lowest_mi = float('inf')
    best_frequency = 0
    
    for candidate in all_candidates:
        # Compute MI for this candidate
        mi = compute_true_mutual_information(ori_ans_list, aug_ans_list, candidate)
        
        # Total frequency for tie-breaking
        n0 = sum(1 for ans in ori_ans_list if ans == candidate)
        n1 = sum(1 for ans in aug_ans_list if ans == candidate) 
        total_freq = n0 + n1
        
        # Select candidate with lowest MI (most source-agnostic)
        # Ties broken by higher frequency
        if mi < lowest_mi or (abs(mi - lowest_mi) < 1e-10 and total_freq > best_frequency):
            best_answer = candidate
            lowest_mi = mi
            best_frequency = total_freq
    
    return best_answer


def determinant_mutual_info_score(ori_ans_list, aug_ans_list, empty_value=''):
    """Select answer with determinant closest to zero (most source-agnostic).
    
    For each candidate answer a, build 2x2 contingency table:
                    S=0     S=1
    Ya=1 (a)        n0      n1  
    Ya=0 (≠a)       N0-n0   N1-n1
    
    Compute determinant: det(a) = n0×(N1-n1) - n1×(N0-n0)
    Choose answer with |det(a)| closest to 0 (most balanced across sources).
    
    Args:
        ori_ans_list: List of answers from original branch (S=0)
        aug_ans_list: List of answers from augmented branch (S=1)
        empty_value: Return value if no valid answer found
        
    Returns:
        Answer with |determinant| closest to 0, ties broken by frequency
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    if not ori_ans_list and not aug_ans_list:
        return empty_value
    
    # Get all unique candidate answers
    all_candidates = set(ori_ans_list + aug_ans_list)
    
    if not all_candidates:
        return empty_value
        
    N0 = len(ori_ans_list)  # Total ori samples
    N1 = len(aug_ans_list)  # Total aug samples
    
    best_answer = empty_value
    min_abs_determinant = float('inf')
    best_frequency = 0
    
    for candidate in all_candidates:
        # Count occurrences in each source
        n0 = sum(1 for ans in ori_ans_list if ans == candidate)
        n1 = sum(1 for ans in aug_ans_list if ans == candidate)
        
        # Compute determinant of 2x2 contingency table
        # det = n0×(N1-n1) - n1×(N0-n0)
        det = n0 * (N1 - n1) - n1 * (N0 - n0)
        abs_det = abs(det)
        
        # Total frequency for tie-breaking
        total_freq = n0 + n1
        
        # Select candidate with smallest |determinant| (most source-balanced)
        # Ties broken by higher frequency
        if abs_det < min_abs_determinant or (abs_det == min_abs_determinant and total_freq > best_frequency):
            best_answer = candidate
            min_abs_determinant = abs_det
            best_frequency = total_freq
    
    return best_answer


def majority_vote_score(ori_ans_list, aug_ans_list, empty_value=''):
    """Select answer with highest frequency from combined ori+aug branches.

    Simple majority voting: combines all answers from both branches and
    returns the most frequent one. This is the classic majority voting approach.

    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch
        empty_value: Return value if no valid answer found

    Returns:
        Answer with highest total frequency across both branches
    """
    from collections import Counter

    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']

    # Combine all answers from both branches
    all_answers = ori_ans_list + aug_ans_list

    if not all_answers:
        return empty_value

    # Count frequencies and return most common
    counter = Counter(all_answers)
    most_common_answer, _ = counter.most_common(1)[0]

    return most_common_answer


def harmonic_mean_score(ori_ans_list, aug_ans_list, empty_value='', laplace_smoothing=False):
    """Select answer with highest harmonic mean score.
    
    Score(c) = n_x(c) * n_x'(c) / (n_x(c) + n_x'(c)) / 4
    
    Where:
    - n_x(c) = count of answer c in original branch
    - n_x'(c) = count of answer c in augmented branch
    - Division by 4 for normalization
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch
        empty_value: Return value if no valid answer found
        laplace_smoothing: Add +1 to both count occurrences for hard examples
        
    Returns:
        Answer with highest harmonic mean score, ties broken by frequency
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    if not ori_ans_list and not aug_ans_list:
        return empty_value
    
    # Get all unique candidate answers
    all_candidates = set(ori_ans_list + aug_ans_list)
    
    if not all_candidates:
        return empty_value
    
    best_answer = empty_value
    max_score = -1.0
    best_frequency = 0
    
    for candidate in all_candidates:
        # Count occurrences in each source
        n_ori = sum(1 for ans in ori_ans_list if ans == candidate)
        n_aug = sum(1 for ans in aug_ans_list if ans == candidate)
        
        # Apply Laplace smoothing if enabled
        if laplace_smoothing:
            n_ori += 1
            n_aug += 1
        else:
            # Skip if answer doesn't appear in both branches (original behavior)
            if n_ori == 0 or n_aug == 0:
                continue
            
        # Compute harmonic mean score: n_x(c) * n_x'(c) / (n_x(c) + n_x'(c)) / 4
        score = (n_ori * n_aug) / (n_ori + n_aug) / 4.0
        
        # Total frequency for tie-breaking
        total_freq = n_ori + n_aug
        
        # Select candidate with highest score
        # Ties broken by higher frequency
        if score > max_score or (score == max_score and total_freq > best_frequency):
            best_answer = candidate
            max_score = score
            best_frequency = total_freq
    
    return best_answer


def compute_harmonic_mean_score_for_answer(ori_ans_list, aug_ans_list, target_answer, laplace_smoothing=False):
    """Compute harmonic mean score for a specific answer (using math equivalence).
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch  
        target_answer: The answer to compute harmonic mean score for
        laplace_smoothing: Add +1 to both count occurrences for hard examples
        
    Returns:
        Harmonic mean score for the target answer
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    # Count mathematically equivalent occurrences of target answer in each source
    n_ori = count_math_equivalent(ori_ans_list, target_answer)
    n_aug = count_math_equivalent(aug_ans_list, target_answer)
    
    # Apply Laplace smoothing if enabled
    if laplace_smoothing:
        n_ori += 1
        n_aug += 1
    else:
        # Return 0 if answer doesn't appear in both branches (original behavior)
        if n_ori == 0 or n_aug == 0:
            return 0.0
        
    # Compute harmonic mean score: n_x(c) * n_x'(c) / (n_x(c) + n_x'(c)) / 4
    score = (n_ori * n_aug) / (n_ori + n_aug) / 4.0
    return score


def get_unique_math_candidates(ori_ans_list, aug_ans_list):
    """Get unique mathematical candidates from both answer lists.
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch
        
    Returns:
        list: List of unique mathematical candidates (representatives)
    """
    all_answers = ori_ans_list + aug_ans_list
    groups = group_math_equivalent_answers(all_answers)
    return list(groups.keys())


def compute_harmonic_mean_scores_for_uid(ori_ans_list, aug_ans_list, laplace_smoothing=False):
    """Compute harmonic mean scores for all candidates for a given UID (using math equivalence).
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch
        laplace_smoothing: Add +1 to both count occurrences for hard examples
        
    Returns:
        Dict mapping candidate answers to their harmonic mean scores, sorted by score desc
    """
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    if not ori_ans_list or not aug_ans_list:
        return {}
    
    # Get unique mathematical candidates
    unique_candidates = get_unique_math_candidates(ori_ans_list, aug_ans_list)
    
    if not unique_candidates:
        return {}
    
    # Compute harmonic mean scores for all unique candidates
    candidate_scores = []
    for candidate in unique_candidates:
        score = compute_harmonic_mean_score_for_answer(ori_ans_list, aug_ans_list, candidate, laplace_smoothing)
        candidate_scores.append((candidate, score))
    
    # Sort candidates by harmonic mean score (descending)
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    
    return dict(candidate_scores)


def compute_soft_reward_for_answer_with_cached_scores(candidate_scores, target_answer, top_k=8):
    """Compute soft reward for a specific answer using pre-computed harmonic mean scores.
    
    For the top-k candidates ranked by harmonic mean score, rewards are distributed
    evenly from 1.0 to 0.0 with common difference:
    - 1st place: 1.0
    - 2nd place: 1 - 1/(top_k-1)
    - 3rd place: 1 - 2/(top_k-1)
    - ...
    - top_k place: 1 - (top_k-1)/(top_k-1) = 0.0
    - Beyond top_k: 0.0
    
    Args:
        candidate_scores: Dict mapping candidates to harmonic mean scores (sorted desc)
        target_answer: The answer to compute soft reward for
        top_k: Number of top candidates to consider (default 8)
        
    Returns:
        Soft reward value between 0.0 and 1.0
    """
    if not candidate_scores:
        return 0.0
    
    # Get sorted candidates by score (descending)
    sorted_candidates = list(candidate_scores.keys())
    
    # Find the rank of target_answer using mathematical equivalence
    rank = -1
    for i, candidate in enumerate(sorted_candidates):
        if math_equivalent(target_answer, candidate):
            rank = i
            break
    
    if rank == -1:
        return 0.0  # Not found
    
    if rank < top_k:
        # Compute reward with common difference
        if top_k == 1:
            return 1.0
        else:
            reward = 1.0 - rank / (top_k - 1)
            return max(0.0, reward)  # Ensure non-negative
    else:
        return 0.0  # Beyond top_k


def compute_soft_reward_for_answer(ori_ans_list, aug_ans_list, target_answer, top_k=8, laplace_smoothing=False):
    """Compute soft reward for a specific answer based on its harmonic mean rank.
    
    For the top-k candidates ranked by harmonic mean score, rewards are distributed
    evenly from 1.0 to 0.0 with common difference:
    - 1st place: 1.0
    - 2nd place: 1 - 1/(top_k-1)
    - 3rd place: 1 - 2/(top_k-1)
    - ...
    - top_k place: 1 - (top_k-1)/(top_k-1) = 0.0
    - Beyond top_k: 0.0
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch
        target_answer: The answer to compute soft reward for
        top_k: Number of top candidates to consider (default 8)
        laplace_smoothing: Add +1 to both count occurrences for hard examples
        
    Returns:
        Soft reward value between 0.0 and 1.0
    """
    # Compute candidate scores and use cached version
    candidate_scores = compute_harmonic_mean_scores_for_uid(ori_ans_list, aug_ans_list, laplace_smoothing)
    return compute_soft_reward_for_answer_with_cached_scores(candidate_scores, target_answer, top_k)


def compute_average_harmonic_mean_baseline(ori_uid2answers, aug_uid2answers, laplace_smoothing=False):
    """Compute average harmonic mean scores for ori and aug branches as baselines.
    
    Args:
        ori_uid2answers: Dict mapping UIDs to ori answer lists
        aug_uid2answers: Dict mapping UIDs to aug answer lists
        laplace_smoothing: Add +1 to both count occurrences for hard examples
        
    Returns:
        tuple: (ori_baseline, aug_baseline) - average harmonic mean scores
    """
    ori_scores = []
    aug_scores = []
    
    for uid in set(ori_uid2answers.keys()) | set(aug_uid2answers.keys()):
        ori_answers = ori_uid2answers.get(uid, [])
        aug_answers = aug_uid2answers.get(uid, [])
        
        if not ori_answers or not aug_answers:
            continue
            
        # Get all unique answers for this UID
        all_answers = set(ori_answers + aug_answers)
        
        # Compute harmonic mean scores for each answer and track individual answer scores
        for answer in all_answers:
            score = compute_harmonic_mean_score_for_answer(ori_answers, aug_answers, answer, laplace_smoothing)
            if score > 0:  # Only include valid scores
                # Count how many times this answer appears in each branch
                ori_count = sum(1 for ans in ori_answers if ans == answer)  
                aug_count = sum(1 for ans in aug_answers if ans == answer)
                
                # Add score weighted by frequency in each branch
                if ori_count > 0:
                    ori_scores.extend([score] * ori_count)
                if aug_count > 0:
                    aug_scores.extend([score] * aug_count)
    
    ori_baseline = np.mean(ori_scores) if ori_scores else 0.0
    aug_baseline = np.mean(aug_scores) if aug_scores else 0.0
    
    return ori_baseline, aug_baseline


def unified_voting_score(ori_ans_list, aug_ans_list, strategy, bootstrap_samples_per_turn=0, bootstrap_turns=1, empty_value='', harmonic_mean_laplace_smoothing=False, mixed_strategy_weights=None):
    """Unified scoring function for all voting strategies with optional bootstrap.
    
    Args:
        ori_ans_list: List of answers from original branch
        aug_ans_list: List of answers from augmented branch  
        strategy: Voting strategy ("cross", "self", "mutual-info", "true-mutual-info", "harmonic-mean", "majority", "mixed")
        bootstrap_samples_per_turn: Number of samples per bootstrap turn (0 = no bootstrap)
        bootstrap_turns: Number of bootstrap turns
        empty_value: Return value if no valid answer found
        harmonic_mean_laplace_smoothing: Add +1 to both count occurrences in harmonic mean calculation
        mixed_strategy_weights: Dict of strategy weights for "mixed" strategy (e.g., {"harmonic-mean": 0.5, "majority": 0.5})
        
    Returns:
        Answer with highest accumulated score
    """
    import random
    from collections import defaultdict
    
    # Filter out empty answers
    ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
    aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
    
    # Get all unique candidate answers based on strategy
    if strategy == "cross":
        candidates = set(aug_ans_list) if aug_ans_list else set()
    elif strategy == "self":
        candidates = set(ori_ans_list) if ori_ans_list else set()
    else:  # mutual-info, true-mutual-info, harmonic-mean
        candidates = set(ori_ans_list + aug_ans_list)
    
    if not candidates:
        return empty_value
    
    candidate_scores = defaultdict(float)
    
    # If bootstrap disabled, use original data once
    if bootstrap_samples_per_turn == 0:
        bootstrap_turns = 1
        use_bootstrap = False
    else:
        use_bootstrap = True
    
    for turn in range(bootstrap_turns):
        # Prepare data for this turn
        if use_bootstrap:
            # Bootstrap sample with replacement
            turn_ori = [random.choice(ori_ans_list) for _ in range(bootstrap_samples_per_turn)] if ori_ans_list else []
            turn_aug = [random.choice(aug_ans_list) for _ in range(bootstrap_samples_per_turn)] if aug_ans_list else []
        else:
            # Use original data
            turn_ori = ori_ans_list
            turn_aug = aug_ans_list
        
        # Compute scores for each candidate based on strategy
        if strategy == "cross":
            # Cross-voting: use aug majority as pseudo-label
            if turn_aug:
                aug_majority, _ = majority_vote_with_count(turn_aug, empty_value=empty_value)
                if aug_majority != empty_value:
                    candidate_scores[aug_majority] += 1.0
                    
        elif strategy == "self": 
            # Self-voting: use ori majority as pseudo-label
            if turn_ori:
                ori_majority, _ = majority_vote_with_count(turn_ori, empty_value=empty_value)
                if ori_majority != empty_value:
                    candidate_scores[ori_majority] += 1.0
                    
        elif strategy == "mutual-info":
            # F-score based mutual information
            best_candidate = mutual_info_score(turn_ori, turn_aug, empty_value=empty_value)
            if best_candidate != empty_value:
                candidate_scores[best_candidate] += 1.0
                
        elif strategy == "true-mutual-info":
            # True mutual information calculation
            best_candidate = true_mutual_info_score(turn_ori, turn_aug, empty_value=empty_value)
            if best_candidate != empty_value:
                candidate_scores[best_candidate] += 1.0
                
        elif strategy == "harmonic-mean":
            # Harmonic mean scoring
            best_candidate = harmonic_mean_score(turn_ori, turn_aug, empty_value=empty_value, laplace_smoothing=harmonic_mean_laplace_smoothing)
            if best_candidate != empty_value:
                candidate_scores[best_candidate] += 1.0

        elif strategy == "majority":
            # Simple majority voting: count most frequent answer from combined ori+aug
            best_candidate = majority_vote_score(turn_ori, turn_aug, empty_value=empty_value)
            if best_candidate != empty_value:
                candidate_scores[best_candidate] += 1.0

        elif strategy == "mixed":
            # Mixed strategy: randomly choose between strategies with configurable weights
            import random
            import numpy as np

            # Use provided weights or default to 50/50 between harmonic-mean and majority
            if mixed_strategy_weights is None:
                strategy_weights = {"harmonic-mean": 0.5, "majority": 0.5}
            else:
                strategy_weights = mixed_strategy_weights

            strategies = list(strategy_weights.keys())
            weights = list(strategy_weights.values())

            # Normalize weights to sum to 1
            weights = np.array(weights) / np.sum(weights)

            # Use weighted random choice
            chosen_strategy = np.random.choice(strategies, p=weights)

            if chosen_strategy == "harmonic-mean":
                best_candidate = harmonic_mean_score(turn_ori, turn_aug, empty_value=empty_value, laplace_smoothing=harmonic_mean_laplace_smoothing)
            elif chosen_strategy == "majority":
                best_candidate = majority_vote_score(turn_ori, turn_aug, empty_value=empty_value)
            else:
                best_candidate = empty_value

            if best_candidate != empty_value:
                candidate_scores[best_candidate] += 1.0
    
    if not candidate_scores:
        return empty_value
        
    # Return answer with highest accumulated score
    return max(candidate_scores.items(), key=lambda x: x[1])[0]


def compute_wasserstein_1d(p, q):
    """Compute 1D Wasserstein distance between two discrete distributions."""
    # Convert answer lists to probability distributions
    p_counts = Counter(p)
    q_counts = Counter(q)
    
    # Get all unique answers
    all_answers = set(p_counts.keys()) | set(q_counts.keys())
    if len(all_answers) <= 1:
        return 0.0
        
    # Convert to sorted lists for 1D Wasserstein calculation
    sorted_answers = sorted(all_answers)
    p_probs = [p_counts.get(ans, 0) / len(p) for ans in sorted_answers]
    q_probs = [q_counts.get(ans, 0) / len(q) for ans in sorted_answers]
    
    # Compute cumulative distributions
    p_cum = np.cumsum(p_probs)
    q_cum = np.cumsum(q_probs)
    
    # 1D Wasserstein distance is the integral of |F1 - F2|
    return np.sum(np.abs(p_cum - q_cum))


def create_fused_distribution(ans_list_1, ans_list_2):
    """Create fused distribution from two answer lists."""
    # Combine all answers
    fused_answers = ans_list_1 + ans_list_2
    
    # Create probability distribution
    if not fused_answers:
        return {}
    
    fused_counts = Counter(fused_answers)
    total = len(fused_answers)
    fused_dist = {ans: count / total for ans, count in fused_counts.items()}
    
    return fused_dist


def compute_w1_to_fused(ans_list, fused_dist):
    """Compute W1 (Wasserstein-1) distance from answer distribution to fused distribution."""
    if not ans_list or not fused_dist:
        return 0.0
        
    ans_counts = Counter(ans_list)
    total = len(ans_list)
    
    # Get all unique answers from both distributions
    all_answers = set(ans_counts.keys()) | set(fused_dist.keys())
    
    # Convert to probability vectors
    p_probs = []
    q_probs = []
    for ans in sorted(all_answers):  # Sort for consistency
        p_probs.append(ans_counts.get(ans, 0) / total)
        q_probs.append(fused_dist.get(ans, 0))
    
    # For discrete distributions on equal support, W1 = sum of absolute differences of cumulative distributions
    p_vec = np.array(p_probs)
    q_vec = np.array(q_probs)
    
    # Compute cumulative distributions
    p_cum = np.cumsum(p_vec)
    q_cum = np.cumsum(q_vec)
    
    # W1 is the integral (sum) of absolute differences
    w1_dist = np.sum(np.abs(p_cum - q_cum))
    return w1_dist


def compute_w1_efficient(ans_list, fused_dist):
    """Alternative W1 implementation using sample-based approach (more efficient)."""
    if not ans_list or not fused_dist:
        return 0.0
    
    # Create samples based on probability distributions
    all_answers = sorted(set(ans_list) | set(fused_dist.keys()))
    answer_to_idx = {ans: i for i, ans in enumerate(all_answers)}
    
    # Convert ans_list to indices
    samples_a = np.array([answer_to_idx[ans] for ans in ans_list])
    
    # Create samples from fused distribution
    fused_samples = []
    total_fused = sum(fused_dist.values())
    for ans, prob in fused_dist.items():
        count = int(prob * total_fused * len(ans_list))  # Scale to same size
        fused_samples.extend([answer_to_idx[ans]] * count)
    
    if not fused_samples:
        return 0.0
        
    samples_b = np.array(fused_samples[:len(ans_list)])  # Ensure same length
    
    # Use your efficient W1 computation
    if len(samples_b) == 0:
        return 0.0
    
    sorted_a = np.sort(samples_a)
    sorted_b = np.sort(samples_b)
    
    # Ensure same length for comparison
    min_len = min(len(sorted_a), len(sorted_b))
    return np.mean(np.abs(sorted_a[:min_len] - sorted_b[:min_len]))


def compute_w2_to_fused(ans_list, fused_dist):
    """Compute W2 (squared Wasserstein-2) distance from answer distribution to fused distribution."""
    if not ans_list or not fused_dist:
        return 0.0
        
    ans_counts = Counter(ans_list)
    total = len(ans_list)
    
    # Get all unique answers from both distributions
    all_answers = set(ans_counts.keys()) | set(fused_dist.keys())
    
    # Convert to probability vectors
    p_probs = []
    q_probs = []
    for ans in sorted(all_answers):  # Sort for consistency
        p_probs.append(ans_counts.get(ans, 0) / total)
        q_probs.append(fused_dist.get(ans, 0))
    
    # For discrete distributions with equal support, W2² = ||p - q||²
    # This is much simpler and more efficient than solving optimal transport
    p_vec = np.array(p_probs)
    q_vec = np.array(q_probs)
    
    w2_squared = np.sum((p_vec - q_vec) ** 2)
    return w2_squared


def compute_kl_to_fused(ans_list, fused_dist):
    """Compute KL divergence from answer distribution to fused distribution."""
    if not ans_list or not fused_dist:
        return 0.0
        
    ans_counts = Counter(ans_list)
    total = len(ans_list)
    
    kl_div = 0.0
    for ans, count in ans_counts.items():
        p = count / total
        q = fused_dist.get(ans, 1e-8)  # Small epsilon to avoid log(0)
        kl_div += p * np.log(p / q)
    
    return kl_div


def compute_format_compliance_score(response_str):
    """Compute format compliance score for the three required sections.
    
    Expected format:
    STRATEGY: [Name of the chosen strategy]
    REWRITTEN PROBLEM: [Your transformed problem statement.]
    SOLUTION: [Your step-by-step solution...]
    
    Returns:
        dict with scores for each section and total score:
        {
            'strategy_score': float (0.0 or 1.0),
            'rewritten_score': float (0.0 or 1.0), 
            'solution_score': float (0.0 or 1.0),
            'total_score': float (0.0 to 3.0),
            'rewritten_text': str (extracted text or empty)
        }
    """
    import re
    
    result = {
        'strategy_score': 0.0,
        'rewritten_score': 0.0,
        'solution_score': 0.0,
        'total_score': 0.0,
        'rewritten_text': ''
    }
    
    # Define patterns for each section
    strategy_pattern = r'STRATEGY:\s*(.+?)(?=\n|\r|REWRITTEN PROBLEM:|$)'
    rewritten_pattern = r'REWRITTEN PROBLEM:\s*(.*?)\s*(?:SOLUTION:|$)'
    solution_pattern = r'SOLUTION:\s*(.*?)(?:$)'
    
    # Define placeholder patterns to reject
    placeholder_patterns = [
        r'^\[.*\]$',  # [Some placeholder text]
        r'^\.+$',     # Just dots
        r'^-+$',      # Just dashes
        r'^.*placeholder.*$',  # Contains "placeholder"
        r'^.*\[.*\].*$',  # Contains brackets (likely placeholder)
    ]
    
    def is_valid_content(text):
        """Check if text is valid (non-empty and not placeholder)"""
        if not text or len(text.strip()) == 0:
            return False
        for pattern in placeholder_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return False
        return True
    
    # Check STRATEGY section
    strategy_match = re.search(strategy_pattern, response_str, re.DOTALL | re.IGNORECASE)
    if strategy_match:
        strategy_text = strategy_match.group(1).strip()
        if is_valid_content(strategy_text):
            result['strategy_score'] = 1.0
    
    # Check REWRITTEN PROBLEM section
    rewritten_match = re.search(rewritten_pattern, response_str, re.DOTALL | re.IGNORECASE)
    if rewritten_match:
        rewritten_text = rewritten_match.group(1).strip()
        if is_valid_content(rewritten_text):
            result['rewritten_score'] = 1.0
            result['rewritten_text'] = rewritten_text
    
    # Check SOLUTION section
    solution_match = re.search(solution_pattern, response_str, re.DOTALL | re.IGNORECASE)
    if solution_match:
        solution_text = solution_match.group(1).strip()
        if is_valid_content(solution_text):
            result['solution_score'] = 1.0
    
    # Calculate total score
    result['total_score'] = result['strategy_score'] + result['rewritten_score'] + result['solution_score']
    
    return result


def extract_rewritten_problem(response_str):
    """Extract the rewritten problem text from response string (backward compatibility).
    
    Returns empty string if format is not followed or content is empty.
    This function maintains backward compatibility with existing code.
    """
    format_scores = compute_format_compliance_score(response_str)
    return format_scores['rewritten_text'] if format_scores['total_score'] == 3.0 else ""


def compute_sharpness_objective(answer_matches_pseudo_count, target_matches=4, max_rollouts=8):
    """Compute sharpness objective: |#matches - target|^2 / (max_rollouts^2/4)
    
    Args:
        answer_matches_pseudo_count: Number of answers that match pseudo label
        target_matches: Target number of matches (default 4 for balanced sharpness)
        max_rollouts: Maximum number of rollouts (default 8)
    
    Returns:
        Normalized sharpness penalty in [0, 1], where 0 = perfect, 1 = worst
    """
    deviation = abs(answer_matches_pseudo_count - target_matches)
    sharpness_penalty = (deviation ** 2) / (max_rollouts ** 2 / 4)  # Use (max_rollouts^2)/4 for better normalization
    return min(1.0, sharpness_penalty)  # Clamp to [0, 1]


def compute_js_divergence_consistency_penalty(ori_answers, aug_answers):
    """Compute Jensen-Shannon divergence between ori and aug answer distributions.
    
    D_JS = 0.5 * KL(p0 || m) + 0.5 * KL(p1 || m), where m = (p0 + p1)/2
    
    Args:
        ori_answers: List of answers from original branch
        aug_answers: List of answers from augmented branch
    
    Returns:
        JS divergence value in [0, 1], where 0 = identical distributions, 1 = maximally different
        We want high JS divergence (different distributions), so we return 1 - JS as penalty
    """
    if not ori_answers or not aug_answers:
        return 0.0  # No penalty if either list is empty
    
    # Merge candidate sets to ensure same support
    all_candidates = set(ori_answers + aug_answers)
    if len(all_candidates) <= 1:
        return 0.0  # No penalty if only one unique answer
    
    # Convert to probability distributions over merged candidate set
    ori_counts = Counter(ori_answers)
    aug_counts = Counter(aug_answers)
    
    p0 = np.array([ori_counts.get(ans, 0) / len(ori_answers) for ans in sorted(all_candidates)])
    p1 = np.array([aug_counts.get(ans, 0) / len(aug_answers) for ans in sorted(all_candidates)])
    
    # Compute mixture distribution m = (p0 + p1) / 2
    m = (p0 + p1) / 2
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    p0 = p0 + epsilon
    p1 = p1 + epsilon
    m = m + epsilon
    
    # Renormalize after adding epsilon
    p0 = p0 / np.sum(p0)
    p1 = p1 / np.sum(p1)
    m = m / np.sum(m)
    
    # Compute KL divergences
    kl_p0_m = np.sum(p0 * np.log(p0 / m))
    kl_p1_m = np.sum(p1 * np.log(p1 / m))
    
    # Jensen-Shannon divergence
    js_div = 0.5 * kl_p0_m + 0.5 * kl_p1_m
    
    # JS divergence is bounded by log(2) ≈ 0.693, normalize to [0, 1]
    normalized_js = js_div / np.log(2)
    
    # We want distributions to be DIFFERENT, so we return JS divergence directly as reward
    # Higher JS divergence = more different = less penalty
    consistency_penalty = 1.0 - normalized_js  # Convert to penalty: 0 = maximally different, 1 = identical
    
    return max(0.0, min(1.0, consistency_penalty))  # Clamp to [0, 1]


class SelfHarmonyManager:
    """The reward manager.

    Args:
        accuracy_mode: Mode for pseudo label accuracy calculations ("fast", "moderate", "full").
            - "fast": Quick string-based equivalence (rough reference, fastest)
            - "moderate": Normalization-based equivalence (balanced speed/accuracy, default)
            - "full": Full symbolic math equivalence (most accurate, like TTRL)
        mixed_strategy_weights: Dict of strategy weights for "mixed" voting strategy.
            - Default: {"harmonic-mean": 0.5, "majority": 0.5} (50/50 split)
            - Example: {"harmonic-mean": 0.7, "majority": 0.3} (70/30 split)
        reward_combination_mode: How to combine base reward with format and consistency components.
            - "multiplicative": Apply penalties multiplicatively (original behavior, may compound)
            - "additive": Add weighted components linearly (more principled, avoids compounding)
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", ot_loss=None, voting_strategy="cross",
                 random_group_sample_size=8, warmup_steps=0, warmup_voting_strategy="self",
                 format_penalty_coef=0.3, sharpness_objective_coef=0.2, enable_aug_grpo_reward=True,
                 penalty_type="sharpness", bootstrap_samples_per_turn=0, bootstrap_turns=1,
                 use_harmonic_mean_reward=False, use_soft_reward=False, use_math_equivalence=True,
                 harmonic_mean_laplace_smoothing=False, choice_type="math", choice_size=4,
                 accuracy_mode="moderate", mixed_strategy_weights=None, reward_combination_mode="multiplicative") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        # ot_loss can be: None/False (majority voting), "w1", "w2", "kl", True (backward compatibility -> "w2")
        if ot_loss is True:
            self.ot_loss = "w2"  # Default for backward compatibility
        elif ot_loss in ["w1", "w2", "kl"]:
            self.ot_loss = ot_loss
        else:
            self.ot_loss = None  # Use majority voting
        # voting_strategy can be: "cross", "self", "mutual-info", "true-mutual-info", "harmonic-mean", "majority", "mixed"
        self.voting_strategy = voting_strategy if voting_strategy in ["cross", "self", "mutual-info", "true-mutual-info", "harmonic-mean", "majority", "mixed"] else "cross"
        # warmup settings for dynamic generation format learning
        self.warmup_steps = warmup_steps
        self.warmup_voting_strategy = warmup_voting_strategy if warmup_voting_strategy in ["cross", "self", "mutual-info", "true-mutual-info", "harmonic-mean", "majority", "mixed"] else "self"
        self.current_step = 0
        self.random_group_sample_size = random_group_sample_size
        self.step_counter = 0  # Track training steps for scheduling
        # New regularization parameters
        self.format_penalty_coef = format_penalty_coef
        self.sharpness_objective_coef = sharpness_objective_coef
        self.enable_aug_grpo_reward = enable_aug_grpo_reward
        self.penalty_type = penalty_type if penalty_type in ["sharpness", "consistency", "none"] else "sharpness"
        self.bootstrap_samples_per_turn = bootstrap_samples_per_turn  # Number of samples per bootstrap turn (0 = disabled)
        self.bootstrap_turns = bootstrap_turns  # Number of bootstrap turns
        self.use_harmonic_mean_reward = use_harmonic_mean_reward  # Enable direct harmonic mean reward (bypasses pseudo labels)
        self.use_soft_reward = use_soft_reward  # Enable soft reward for aug branch (only effective under specific conditions)
        self.use_math_equivalence = use_math_equivalence  # Enable mathematical equivalence for answer comparison (instead of string equality)
        self.harmonic_mean_laplace_smoothing = harmonic_mean_laplace_smoothing  # Add +1 to both count occurrences in harmonic mean calculation
        self.choice_type = choice_type  # Type of comparison: "math" or "choice"
        self.choice_size = choice_size  # Number of choices for multiple choice questions (4 for A-D, 8 for A-H, etc.)
        self.accuracy_mode = accuracy_mode if accuracy_mode in ["fast", "moderate", "full"] else "moderate"  # Mode for pseudo label accuracy calculations
        # Mixed strategy weights: default 50/50 between harmonic-mean and majority
        if mixed_strategy_weights is None:
            self.mixed_strategy_weights = {"harmonic-mean": 0.5, "majority": 0.5}
        else:
            self.mixed_strategy_weights = mixed_strategy_weights
        # Reward combination mode: "multiplicative" or "additive"
        self.reward_combination_mode = reward_combination_mode if reward_combination_mode in ["multiplicative", "additive"] else "multiplicative"

    def _combine_rewards(self, base_reward, format_score=None, consistency_penalty=None):
        """Combine base reward with format and consistency components using the configured mode.

        Args:
            base_reward: Base reward value [0.0, 1.0]
            format_score: Format compliance score [0.0, 1.0] (higher = better)
            consistency_penalty: Consistency penalty [0.0, 1.0] (higher = more penalty)

        Returns:
            Combined reward value
        """
        if self.reward_combination_mode == "additive":
            # Additive combination with weights
            combined_reward = base_reward

            # Add format component (convert score to reward contribution)
            if format_score is not None:
                format_reward = format_score * self.format_penalty_coef
                combined_reward += format_reward

            # Add consistency component (convert penalty to reward contribution)
            if consistency_penalty is not None:
                if self.sharpness_objective_coef < 0:
                    # Negative coefficient: reward consistency
                    consistency_reward = consistency_penalty * abs(self.sharpness_objective_coef)
                    combined_reward += consistency_reward
                else:
                    # Positive coefficient: penalize consistency (reward diversity)
                    diversity_reward = (1.0 - consistency_penalty) * self.sharpness_objective_coef
                    combined_reward += diversity_reward

            return combined_reward

        else:  # multiplicative (original behavior)
            combined_reward = base_reward

            # Apply format penalty multiplicatively
            if format_score is not None:
                format_penalty = (1.0 - format_score) * self.format_penalty_coef
                combined_reward *= (1.0 - format_penalty)

            # Apply consistency penalty multiplicatively
            if consistency_penalty is not None:
                if self.sharpness_objective_coef < 0:
                    # Negative coefficient: reward consistency
                    effective_penalty = 1.0 - consistency_penalty
                    combined_reward *= (1.0 + abs(self.sharpness_objective_coef) * effective_penalty)
                else:
                    # Positive coefficient: penalize consistency
                    combined_reward *= (1.0 - self.sharpness_objective_coef * consistency_penalty)

            return combined_reward

    def _compare_answers(self, answer1, answer2):
        """Compare two answers using TTRL's moderate equivalence for reward assignment.

        Uses moderate equivalence (normalization-based) which provides a good balance
        between speed and accuracy for reward computation, following TTRL's approach.

        Args:
            answer1: First answer to compare
            answer2: Second answer to compare

        Returns:
            bool: True if answers are considered equivalent
        """
        if self.use_math_equivalence:
            # Use moderate equivalence for reward assignment (TTRL approach)
            return math_equivalent_with_choice(answer1, answer2, self.choice_type, self.choice_size, mode='moderate')
        else:
            # Simple string comparison
            if answer1 is None or answer2 is None:
                return False
            return str(answer1).strip() == str(answer2).strip()

    def _compare_answers_with_math_equivalence(self, answer1, answer2):
        """Compare two answers using mathematical equivalence regardless of use_math_equivalence setting.

        This method is specifically for ground truth accuracy calculations where we always want
        to use mathematical equivalence for more accurate assessment.

        Args:
            answer1: First answer to compare
            answer2: Second answer to compare

        Returns:
            bool: True if answers are mathematically equivalent
        """
        return math_equivalent_with_choice(answer1, answer2, self.choice_type, self.choice_size)

    def _compare_answers_for_accuracy(self, answer1, answer2):
        """Compare two answers for pseudo label accuracy calculations using configurable mode.

        This method allows choosing between different equivalence levels for accuracy calculations:
        - 'fast': Quick string-based equivalence (rough reference)
        - 'moderate': Normalization-based equivalence (balanced speed/accuracy)
        - 'full': Full symbolic math equivalence (most accurate, like TTRL)

        Args:
            answer1: First answer to compare
            answer2: Second answer to compare

        Returns:
            bool: True if answers are considered equivalent
        """
        if self.use_math_equivalence:
            return math_equivalent_with_choice(answer1, answer2, self.choice_type, self.choice_size, mode=self.accuracy_mode)
        else:
            # Simple string comparison
            if answer1 is None or answer2 is None:
                return False
            return str(answer1).strip() == str(answer2).strip()

    def _compute_harmonic_mean_score_for_answer(self, ori_ans_list, aug_ans_list, target_answer):
        """Compute harmonic mean score using the configured comparison method."""
        # Filter out empty answers
        ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
        aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
        
        # Count occurrences using configured comparison method
        n_ori = sum(1 for ans in ori_ans_list if self._compare_answers(ans, target_answer))
        n_aug = sum(1 for ans in aug_ans_list if self._compare_answers(ans, target_answer))
        
        # Apply Laplace smoothing if enabled
        if self.harmonic_mean_laplace_smoothing:
            n_ori += 1
            n_aug += 1
        else:
            # Return 0 if answer doesn't appear in both branches (original behavior)
            if n_ori == 0 or n_aug == 0:
                return 0.0
            
        # Compute harmonic mean score: n_x(c) * n_x'(c) / (n_x(c) + n_x'(c)) / 4
        score = (n_ori * n_aug) / (n_ori + n_aug) / len(ori_ans_list) * 2
        return score

    def _compute_harmonic_mean_scores_for_uid(self, ori_ans_list, aug_ans_list):
        """Compute harmonic mean scores for all candidates using configured comparison method."""
        # Filter out empty answers
        ori_ans_list = [a for a in ori_ans_list if a is not None and str(a).strip() != '']
        aug_ans_list = [a for a in aug_ans_list if a is not None and str(a).strip() != '']
        
        if not ori_ans_list or not aug_ans_list:
            return {}
        
        # Get unique candidates using configured comparison method
        if self.use_math_equivalence:
            unique_candidates = get_unique_math_candidates(ori_ans_list, aug_ans_list)
        else:
            # For string comparison, just use set for uniqueness
            unique_candidates = list(set(ori_ans_list + aug_ans_list))
        
        if not unique_candidates:
            return {}
        
        # Compute harmonic mean scores for all unique candidates
        candidate_scores = []
        for candidate in unique_candidates:
            score = self._compute_harmonic_mean_score_for_answer(ori_ans_list, aug_ans_list, candidate)
            candidate_scores.append((candidate, score))
        
        # Sort candidates by harmonic mean score (descending)
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        return dict(candidate_scores)

    def _compute_soft_reward_for_answer_with_cached_scores(self, candidate_scores, target_answer, top_k=8):
        """Compute soft reward using cached scores and configured comparison method."""
        if not candidate_scores:
            return 0.0
        
        # Get sorted candidates by score (descending)
        sorted_candidates = list(candidate_scores.keys())
        
        # Find the rank of target_answer using configured comparison method
        rank = -1
        for i, candidate in enumerate(sorted_candidates):
            if self._compare_answers(target_answer, candidate):
                rank = i
                break
        
        if rank == -1:
            return 0.0  # Not found
        
        if rank < top_k:
            # Compute reward with common difference
            if top_k == 1:
                return 1.0
            else:
                reward = 1.0 - rank / (top_k - 1)
                return max(0.0, reward)  # Ensure non-negative
        else:
            return 0.0  # Beyond top_k

    def _get_effective_voting_strategy(self):
        """Get the current effective voting strategy based on warmup status."""
        if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
            return self.warmup_voting_strategy
        else:
            return self.voting_strategy

    def _extract_valid_response_str(self, item):
        prompt_ids = item.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_prompt_length = item.batch["attention_mask"][:prompt_length].sum()
        response_ids = item.batch["responses"]
        valid_response_length = item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        return response_str

    def __call__(self, data_ori: DataProto, data_aug: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        assert len(data_ori) == len(data_aug), "The original and augmented data must have the same length."
        
        # Increment step counter and get effective voting strategy
        self.current_step += 1
        effective_strategy = self._get_effective_voting_strategy()
        
        # Log warmup status
        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps + 1:
            if self.current_step <= self.warmup_steps:
                print(f"[WARMUP] Step {self.current_step}/{self.warmup_steps} - Using '{effective_strategy}' voting strategy")
            elif self.current_step == self.warmup_steps + 1:
                print(f"[WARMUP COMPLETE] Step {self.current_step} - Switching to '{self.voting_strategy}' voting strategy")
        ori_uid2answers = defaultdict(list)
        ori_all_answers = []
        for i, item in enumerate(data_ori):
            uid = item.non_tensor_batch["uid"]
            ans = self.extract_answer(self._extract_valid_response_str(item))
            ori_uid2answers[uid].append(ans)
            ori_all_answers.append(ans)
        aug_uid2answers = defaultdict(list)
        aug_all_answers = []
        for j, item in enumerate(data_aug):
            uid = item.non_tensor_batch["uid"]
            response_str = self._extract_valid_response_str(item)
            ans = self.extract_answer(response_str)
            aug_uid2answers[uid].append(ans)
            aug_all_answers.append(ans)
        
        if self.use_harmonic_mean_reward:
            # Use direct harmonic mean reward with baselines 
            print("using direct harmonic mean reward")
            reward_ori, reward_aug, extra_info = self._compute_harmonic_mean_rewards(data_ori, data_aug, ori_uid2answers, aug_uid2answers,
                                                                                   ori_all_answers, aug_all_answers, effective_strategy)
        elif self.ot_loss:
            # Use Wasserstein/OT loss with fused distribution
            print("using OT reward")
            reward_ori, reward_aug, extra_info = self._compute_ot_rewards(data_ori, data_aug, ori_uid2answers, aug_uid2answers, 
                                                                        ori_all_answers, aug_all_answers)
        else:
            # Use original majority voting approach
            print(f"using major voting reward with {effective_strategy} strategy")
            reward_ori, reward_aug, extra_info = self._compute_majority_voting_rewards(data_ori, data_aug, ori_uid2answers, aug_uid2answers,
                                                                                     ori_all_answers, aug_all_answers, effective_strategy)
        
        if return_dict:
            return {
                "reward_tensor": (reward_ori, reward_aug),
                "reward_extra_info": extra_info
            }
        else:
            return reward_ori, reward_aug

    def _compute_majority_voting_rewards(self, data_ori, data_aug, ori_uid2answers, aug_uid2answers,
                                       ori_all_answers, aug_all_answers, voting_strategy=None):
        """Majority voting reward calculation with configurable voting strategy."""
        from collections import defaultdict
        if voting_strategy is None:
            voting_strategy = self.voting_strategy
            
        # Unified voting strategy with optional bootstrap
        ori_uid2pseudo = {}
        aug_uid2pseudo = {}
        
        # Pre-compute harmonic mean scores for soft reward optimization (when needed)
        uid2harmonic_scores = {}
        need_harmonic_scores = (self.use_soft_reward and 
                               voting_strategy == "harmonic-mean" and 
                               not self.use_harmonic_mean_reward and
                               self.enable_aug_grpo_reward)
        
        # For each UID, compute unified score with bootstrap support
        for uid in set(ori_uid2answers.keys()) | set(aug_uid2answers.keys()):
            ori_answers = ori_uid2answers.get(uid, [])
            aug_answers = aug_uid2answers.get(uid, [])
            
            if voting_strategy == "cross":
                # Cross-voting: ori branch uses aug answers, aug branch uses ori answers
                ori_pseudo = unified_voting_score(aug_answers, aug_answers, "self", 
                                                self.bootstrap_samples_per_turn, self.bootstrap_turns, empty_value='',
                                                harmonic_mean_laplace_smoothing=self.harmonic_mean_laplace_smoothing)
                aug_pseudo = unified_voting_score(ori_answers, ori_answers, "self", 
                                                self.bootstrap_samples_per_turn, self.bootstrap_turns, empty_value='',
                                                harmonic_mean_laplace_smoothing=self.harmonic_mean_laplace_smoothing)
                ori_uid2pseudo[uid] = ori_pseudo
                aug_uid2pseudo[uid] = aug_pseudo
            elif voting_strategy == "self":
                # Self-voting: both branches use their own answers
                ori_pseudo = unified_voting_score(ori_answers, ori_answers, "self",
                                                self.bootstrap_samples_per_turn, self.bootstrap_turns, empty_value='',
                                                harmonic_mean_laplace_smoothing=self.harmonic_mean_laplace_smoothing)
                aug_pseudo = unified_voting_score(aug_answers, aug_answers, "self",
                                                self.bootstrap_samples_per_turn, self.bootstrap_turns, empty_value='',
                                                harmonic_mean_laplace_smoothing=self.harmonic_mean_laplace_smoothing)
                ori_uid2pseudo[uid] = ori_pseudo
                aug_uid2pseudo[uid] = aug_pseudo
            else:
                # All other strategies: both branches use same pseudo-label
                pseudo_label = unified_voting_score(ori_answers, aug_answers, voting_strategy,
                                                   self.bootstrap_samples_per_turn, self.bootstrap_turns, empty_value='',
                                                   harmonic_mean_laplace_smoothing=self.harmonic_mean_laplace_smoothing)
                ori_uid2pseudo[uid] = pseudo_label
                aug_uid2pseudo[uid] = pseudo_label
                
                # Pre-compute harmonic mean scores for this UID if needed for soft rewards
                if need_harmonic_scores:
                    uid2harmonic_scores[uid] = self._compute_harmonic_mean_scores_for_uid(ori_answers, aug_answers)
        
        # Within-group pseudo labels with counts (for hit rate analysis)
        ori_uid2self_pseudo = {}
        ori_uid2majority_count = {}
        for uid, ans_list in ori_uid2answers.items():
            pseudo, count = majority_vote_with_count(ans_list, empty_value='', choice_type=self.choice_type, choice_size=self.choice_size)
            ori_uid2self_pseudo[uid] = pseudo
            ori_uid2majority_count[uid] = count
            
        aug_uid2self_pseudo = {}
        aug_uid2majority_count = {}
        for uid, ans_list in aug_uid2answers.items():
            pseudo, count = majority_vote_with_count(ans_list, empty_value='', choice_type=self.choice_type, choice_size=self.choice_size)
            aug_uid2self_pseudo[uid] = pseudo  
            aug_uid2majority_count[uid] = count
 
        reward_tensor_ori = torch.zeros_like(data_ori.batch["responses"], dtype=torch.float32)
        reward_tensor_aug = torch.zeros_like(data_aug.batch["responses"], dtype=torch.float32)

        # Track strategy label accuracy (for debugging)
        mutual_info_ground_truth_correct = 0  # How often strategy pseudo labels match ground truth
        total_with_ground_truth = 0
        processed_uids = set()  # Track which UIDs we've seen

        # Track HMS upper bound metrics (for harmonic-mean strategy debugging)
        ori_upper_bound_correct = 0
        aug_upper_bound_correct = 0
        hms_upper_bound_correct = 0
        
        N = len(data_ori) 
        for i in range(N):
            ori_item = data_ori[i]
            aug_item = data_aug[i]
            uid = ori_item.non_tensor_batch["uid"]
            
            ori_pseudo = ori_uid2pseudo[uid]  # Cross-group pseudo for training
            aug_pseudo = aug_uid2pseudo[uid]  # Cross-group pseudo for training
            ori_self_pseudo = ori_uid2self_pseudo[uid]  # Within-group pseudo for analysis
            aug_self_pseudo = aug_uid2self_pseudo[uid]  # Within-group pseudo for analysis
            
            # Check ground truth correctness if available
            if "reward_model" in ori_item.non_tensor_batch and "ground_truth" in ori_item.non_tensor_batch["reward_model"]:
                ground_truth = ori_item.non_tensor_batch["reward_model"]["ground_truth"]
                total_with_ground_truth += 1
                
                # Compute pseudo label using the CURRENT voting strategy for accuracy comparison
                ori_answers = ori_uid2answers.get(uid, [])
                aug_answers = aug_uid2answers.get(uid, [])

                # Use unified scoring function for strategy accuracy computation
                if voting_strategy == "cross":
                    # For cross-voting, use the pseudo that was actually selected for ori
                    strategy_pseudo = ori_pseudo
                elif voting_strategy == "self":
                    # For self-voting, use ori self-pseudo
                    strategy_pseudo = ori_self_pseudo
                else:
                    # For mutual-info, true-mutual-info, harmonic-mean strategies
                    strategy_pseudo = unified_voting_score(ori_answers, aug_answers, voting_strategy,
                                                          self.bootstrap_samples_per_turn, self.bootstrap_turns, empty_value='',
                                                          harmonic_mean_laplace_smoothing=self.harmonic_mean_laplace_smoothing)

                # Use configurable equivalence mode for accuracy calculations
                mutual_info_correct = self._compare_answers_for_accuracy(strategy_pseudo, ground_truth)

                if mutual_info_correct:
                    mutual_info_ground_truth_correct += 1
                
                # HMS Validation Metrics Computation
                if voting_strategy == "harmonic-mean":
                    # Get all unique candidates from both branches
                    if self.use_math_equivalence:
                        all_candidates = set(ori_answers + aug_answers)
                    else:
                        all_candidates = set(ori_answers + aug_answers)
                    
                    # 1. Upper Bound Analysis - check if ANY candidate matches GT
                    any_ori_candidate_correct = any(self._compare_answers_for_accuracy(ans, ground_truth) for ans in set(ori_answers))
                    any_aug_candidate_correct = any(self._compare_answers_for_accuracy(ans, ground_truth) for ans in set(aug_answers))
                    any_all_candidate_correct = any(self._compare_answers_for_accuracy(ans, ground_truth) for ans in all_candidates)
                    
                    if any_ori_candidate_correct:
                        ori_upper_bound_correct += 1
                    if any_aug_candidate_correct:
                        aug_upper_bound_correct += 1
                    if any_all_candidate_correct:
                        hms_upper_bound_correct += 1
                    
            
            # Original data reward (use cross-group pseudo for training)
            ori_ans = ori_all_answers[i]
            valid_response_length = ori_item.batch["attention_mask"][ori_item.batch["prompts"].shape[-1]:].sum().item()
            ori_reward = 1.0 if self._compare_answers(ori_ans, ori_pseudo) and ori_pseudo != '' else 0.0
            if valid_response_length > 0:
                reward_tensor_ori[i, valid_response_length - 1] = ori_reward
            
            # Augmented data reward (use cross-group pseudo for training)
            aug_ans = aug_all_answers[i]
            valid_response_length = aug_item.batch["attention_mask"][aug_item.batch["prompts"].shape[-1]:].sum().item()
            
            # Check if aug branch GRPO reward is enabled
            if self.enable_aug_grpo_reward:
                # Check if soft reward conditions are met
                if (self.use_soft_reward and 
                    voting_strategy == "harmonic-mean" and 
                    not self.use_harmonic_mean_reward):
                    # Use cached harmonic mean scores for soft reward (optimization)
                    if uid in uid2harmonic_scores:
                        aug_reward = self._compute_soft_reward_for_answer_with_cached_scores(
                            uid2harmonic_scores[uid], aug_ans, top_k=8)
                    else:
                        # Fallback to original computation if cache missed
                        ori_answers = ori_uid2answers.get(uid, [])
                        aug_answers = aug_uid2answers.get(uid, [])
                        aug_reward = compute_soft_reward_for_answer(ori_answers, aug_answers, aug_ans, top_k=8, laplace_smoothing=self.harmonic_mean_laplace_smoothing)
                else:
                    # Use hard reward (original logic with configurable equivalence)
                    aug_reward = 1.0 if self._compare_answers(aug_ans, aug_pseudo) and aug_pseudo != '' else 0.0
            else:
                # Disable GRPO reward - use format compliance as base reward
                aug_reward = 1.0
                
            if valid_response_length > 0:
                reward_tensor_aug[i, valid_response_length - 1] = aug_reward
            
            # Track processed UIDs
            processed_uids.add(uid)
        
        
        # Calculate strategy label accuracy (for debugging)
        strategy_gt_accuracy = mutual_info_ground_truth_correct / total_with_ground_truth if total_with_ground_truth > 0 else 0.0
        
        extra_info = {
            "voting_strategy": voting_strategy,
            "strategy_pseudo_gt_accuracy": strategy_gt_accuracy,  # Accuracy of current voting strategy (for debugging)
            "mutual_info_ground_truth_correct": mutual_info_ground_truth_correct,
            "total_with_ground_truth": total_with_ground_truth,
        }
        
        # Add HMS Upper Bound Metrics (only when using harmonic-mean strategy)
        if voting_strategy == "harmonic-mean" and total_with_ground_truth > 0:
            extra_info.update({
                # Upper Bound Analysis (for debugging)
                "hms_upper_bound_accuracy": hms_upper_bound_correct / total_with_ground_truth,
                "ori_upper_bound_accuracy": ori_upper_bound_correct / total_with_ground_truth,
                "aug_upper_bound_accuracy": aug_upper_bound_correct / total_with_ground_truth,
                "hms_upper_bound_correct": hms_upper_bound_correct,
                "ori_upper_bound_correct": ori_upper_bound_correct,
                "aug_upper_bound_correct": aug_upper_bound_correct,
            })
            
        return reward_tensor_ori, reward_tensor_aug, extra_info

    def _compute_harmonic_mean_rewards(self, data_ori, data_aug, ori_uid2answers, aug_uid2answers,
                                     ori_all_answers, aug_all_answers, voting_strategy=None):
        """Direct harmonic mean reward calculation using average scores as baselines."""
        if voting_strategy is None:
            voting_strategy = self.voting_strategy
            
        # Compute average harmonic mean baselines for ori and aug
        ori_baseline, aug_baseline = compute_average_harmonic_mean_baseline(ori_uid2answers, aug_uid2answers, self.harmonic_mean_laplace_smoothing)
        
        reward_tensor_ori = torch.zeros_like(data_ori.batch["responses"], dtype=torch.float32)
        reward_tensor_aug = torch.zeros_like(data_aug.batch["responses"], dtype=torch.float32)

        # Track metrics
        total_pairs = 0
        ori_above_baseline = 0
        aug_above_baseline = 0
        ori_total_score = 0.0
        aug_total_score = 0.0
        ori_total_reward = 0.0
        aug_total_reward = 0.0
        harmonic_mean_scores_ori = []
        harmonic_mean_scores_aug = []
        rewards_ori = []
        rewards_aug = []

        N = len(data_ori)
        for i in range(N):
            ori_item = data_ori[i]
            aug_item = data_aug[i]
            uid = ori_item.non_tensor_batch["uid"]
            
            ori_answers = ori_uid2answers.get(uid, [])
            aug_answers = aug_uid2answers.get(uid, [])
            
            # Get individual answers for this rollout
            ori_ans = ori_all_answers[i]
            aug_ans = aug_all_answers[i]
            
            # Compute harmonic mean scores for individual answers
            ori_score = self._compute_harmonic_mean_score_for_answer(ori_answers, aug_answers, ori_ans)
            aug_score = self._compute_harmonic_mean_score_for_answer(ori_answers, aug_answers, aug_ans)
            
            # Use raw harmonic mean scores as rewards - GRPO will handle baseline subtraction
            ori_reward = ori_score  # Let GRPO subtract group averages
            aug_reward = aug_score
            
            # Track statistics
            total_pairs += 1
            if ori_score > ori_baseline:
                ori_above_baseline += 1
            if aug_score > aug_baseline:
                aug_above_baseline += 1
            
            ori_total_score += ori_score
            aug_total_score += aug_score
            ori_total_reward += ori_reward
            aug_total_reward += aug_reward
            
            harmonic_mean_scores_ori.append(ori_score)
            harmonic_mean_scores_aug.append(aug_score)
            rewards_ori.append(ori_reward)
            rewards_aug.append(aug_reward)
            
            
            # Set rewards at end of valid response
            ori_valid_length = ori_item.batch["attention_mask"][ori_item.batch["prompts"].shape[-1]:].sum().item()
            if ori_valid_length > 0:
                reward_tensor_ori[i, ori_valid_length - 1] = ori_reward
                
            aug_valid_length = aug_item.batch["attention_mask"][aug_item.batch["prompts"].shape[-1]:].sum().item()
            if aug_valid_length > 0:
                reward_tensor_aug[i, aug_valid_length - 1] = aug_reward

        # Calculate metrics
        ori_above_baseline_rate = ori_above_baseline / total_pairs if total_pairs > 0 else 0.0
        aug_above_baseline_rate = aug_above_baseline / total_pairs if total_pairs > 0 else 0.0
        avg_ori_score = ori_total_score / total_pairs if total_pairs > 0 else 0.0
        avg_aug_score = aug_total_score / total_pairs if total_pairs > 0 else 0.0
        avg_ori_reward = ori_total_reward / total_pairs if total_pairs > 0 else 0.0
        avg_aug_reward = aug_total_reward / total_pairs if total_pairs > 0 else 0.0
        
        extra_info = {
            "reward_mode": "harmonic_mean_direct",
            "voting_strategy": voting_strategy,
            "ori_computed_baseline": ori_baseline,  # For monitoring only - not applied
            "aug_computed_baseline": aug_baseline,  # For monitoring only - not applied
            "ori_above_baseline_rate": ori_above_baseline_rate,
            "aug_above_baseline_rate": aug_above_baseline_rate,
            "ori_above_baseline_count": ori_above_baseline,
            "aug_above_baseline_count": aug_above_baseline,
            "avg_ori_harmonic_score": avg_ori_score,
            "avg_aug_harmonic_score": avg_aug_score,
            "avg_ori_reward": avg_ori_reward,
            "avg_aug_reward": avg_aug_reward,
            "total_pairs": total_pairs,
            "ori_harmonic_score_std": np.std(harmonic_mean_scores_ori) if len(harmonic_mean_scores_ori) > 1 else 0.0,
            "aug_harmonic_score_std": np.std(harmonic_mean_scores_aug) if len(harmonic_mean_scores_aug) > 1 else 0.0,
            "ori_reward_std": np.std(rewards_ori) if len(rewards_ori) > 1 else 0.0,
            "aug_reward_std": np.std(rewards_aug) if len(rewards_aug) > 1 else 0.0,
            # Note: GRPO will apply baseline subtraction using group averages
            "baseline_handling": "GRPO_handles_baseline_subtraction",
        }
        
        return reward_tensor_ori, reward_tensor_aug, extra_info

    def _compute_ot_rewards(self, data_ori, data_aug, ori_uid2answers, aug_uid2answers, 
                          ori_all_answers, aug_all_answers):
        """Wasserstein/OT loss reward calculation with fused distribution."""
        reward_tensor_ori = torch.zeros_like(data_ori.batch["responses"], dtype=torch.float32)
        reward_tensor_aug = torch.zeros_like(data_aug.batch["responses"], dtype=torch.float32)

        # Also compute majority voting for consistency comparison
        ori_uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in aug_uid2answers.items()}
        aug_uid2pseudo = {uid: majority_vote(ans_list, empty_value='') for uid, ans_list in ori_uid2answers.items()}
        
        # Track consistency metrics
        total_pairs = 0
        consistent_pairs = 0
        both_have_pseudo = 0
        total_distance = 0.0
        ori_hit_pseudo = 0  # How often ori answers match their pseudo labels  
        aug_hit_pseudo = 0  # How often aug answers match their pseudo labels
        
        N = len(data_ori)
        for i in range(N):
            # Get UID and corresponding answer lists
            ori_item = data_ori[i]
            aug_item = data_aug[i]
            uid = ori_item.non_tensor_batch["uid"]
            
            ori_ans_list = ori_uid2answers[uid]
            aug_ans_list = aug_uid2answers[uid]
            
            # Track majority voting consistency for comparison
            ori_pseudo = ori_uid2pseudo[uid]
            aug_pseudo = aug_uid2pseudo[uid]
            if ori_pseudo != '' and aug_pseudo != '':
                both_have_pseudo += 1
                if self._compare_answers(ori_pseudo, aug_pseudo):
                    consistent_pairs += 1
            total_pairs += 1
            
            # Check how often individual answers hit majority vote
            ori_ans = ori_all_answers[i]
            aug_ans = aug_all_answers[i]
            if ori_ans == ori_pseudo and ori_pseudo != '':
                ori_hit_pseudo += 1
            if aug_ans == aug_pseudo and aug_pseudo != '':
                aug_hit_pseudo += 1
            
            # Create fused distribution P_fused = 1/2G * (sum(delta_ans_yi) + sum(delta_ans_yi'))
            fused_dist = create_fused_distribution(ori_ans_list, aug_ans_list)
            
            # Compute distance to fused distribution based on selected metric
            if self.ot_loss == "w1":
                ori_w_dist = compute_w1_to_fused(ori_ans_list, fused_dist) if ori_ans_list else 0.0
                aug_w_dist = compute_w1_to_fused(aug_ans_list, fused_dist) if aug_ans_list else 0.0
            elif self.ot_loss == "w2":
                ori_w_dist = compute_w2_to_fused(ori_ans_list, fused_dist) if ori_ans_list else 0.0
                aug_w_dist = compute_w2_to_fused(aug_ans_list, fused_dist) if aug_ans_list else 0.0
            elif self.ot_loss == "kl":
                ori_w_dist = compute_kl_to_fused(ori_ans_list, fused_dist) if ori_ans_list else 0.0
                aug_w_dist = compute_kl_to_fused(aug_ans_list, fused_dist) if aug_ans_list else 0.0
            else:
                # Fallback to W2 for safety
                ori_w_dist = compute_w2_to_fused(ori_ans_list, fused_dist) if ori_ans_list else 0.0
                aug_w_dist = compute_w2_to_fused(aug_ans_list, fused_dist) if aug_ans_list else 0.0
            
            # Track average distance for monitoring
            total_distance += (ori_w_dist + aug_w_dist) / 2
            
            # Convert distances to rewards (negative distance, normalized)
            # Reward is higher when distance to fused distribution is lower
            ori_reward = max(0.0, 1.0 - ori_w_dist)  # Clamp to [0, 1]
            aug_reward = max(0.0, 1.0 - aug_w_dist)
            
            # Set rewards at the end of valid response
            ori_valid_length = ori_item.batch["attention_mask"][ori_item.batch["prompts"].shape[-1]:].sum().item()
            if ori_valid_length > 0:
                reward_tensor_ori[i, ori_valid_length - 1] = ori_reward
                
            aug_valid_length = aug_item.batch["attention_mask"][aug_item.batch["prompts"].shape[-1]:].sum().item()
            if aug_valid_length > 0:
                reward_tensor_aug[i, aug_valid_length - 1] = aug_reward
        
        # Calculate consistency metrics (for comparison with majority voting)
        consistency_rate = consistent_pairs / both_have_pseudo if both_have_pseudo > 0 else 0.0
        pseudo_coverage = both_have_pseudo / total_pairs if total_pairs > 0 else 0.0
        avg_distance = total_distance / N if N > 0 else 0.0
        ori_hit_rate = ori_hit_pseudo / total_pairs if total_pairs > 0 else 0.0
        aug_hit_rate = aug_hit_pseudo / total_pairs if total_pairs > 0 else 0.0
        
        extra_info = {
            "majority_vote_consistency": consistency_rate,  # For comparison
            "pseudo_label_coverage": pseudo_coverage,
            "consistent_pairs": consistent_pairs,
            "total_valid_pairs": both_have_pseudo,
            "total_pairs": total_pairs,
            "ori_hit_pseudo_rate": ori_hit_rate,
            "aug_hit_pseudo_rate": aug_hit_rate,
            "ori_hit_pseudo_count": ori_hit_pseudo,
            "aug_hit_pseudo_count": aug_hit_pseudo,
            f"avg_{self.ot_loss}_distance": avg_distance,
            "ot_method": self.ot_loss,
        }
            
        return reward_tensor_ori, reward_tensor_aug, extra_info

    def extract_answer(self, solution_str: str):
        from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed
        answer = ''
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
        except Exception as e:
            print(e)
        return answer
