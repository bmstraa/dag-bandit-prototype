# src/verifier.py
import re
import unicodedata
import numpy as np
from typing import List, Set, Tuple

def normalize_text(text: str) -> str:
    """
    Robust text normaliser for factual verification.
    Strips accents, LaTeX, commas from numbers, and punctuation.
    """
    text = text.lower().strip()
    text = re.sub(r'(\d),(\d)', r'\1\2', text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_tokens(text: str) -> Set[str]:
    """
    Lightweight token normalisation for semantic similarity.
    Strips LaTeX, removes punctuation, and splits into words.
    """
    text = text.lower()
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\\\[.*?\\\]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return set(text.split())

def verify_with_rules(model_answer: str, ground_truth: str) -> Tuple[bool, float]:
    """
    Deterministic factual verifier.
    Returns (correct: bool, confidence: float), where confidence is 1.0 if correct else 0.0.
    """
    norm_answer = normalize_text(model_answer)
    norm_truth = normalize_text(ground_truth)

    # Special case: chemical symbols
    if "h2o" in norm_truth or "water" in norm_truth:
        if "h2o" in norm_answer or "water" in norm_answer:
            return True, 1.0

    if norm_truth in norm_answer:
        return True, 1.0

    return False, 0.0

def evaluate_answer(model_output: str, ground_truth: str) -> bool:
    """
    Unified evaluator for final accuracy.
    Uses the same normalisation as the verifier.
    """
    norm_output = normalize_text(model_output)
    norm_truth = normalize_text(ground_truth)

    if "h2o" in norm_truth or "water" in norm_truth:
        if "h2o" in norm_output or "water" in norm_output:
            return True

    return norm_truth in norm_output

def calculate_jaccard_consistency(responses: List[str]) -> float:
    """
    Calculates average pairwise Jaccard similarity on normalised token sets.
    Returns 1.0 if all identical, 0.0 if completely disjoint.
    """
    if len(responses) < 2:
        return 1.0

    token_sets = [normalize_tokens(r) for r in responses]
    similarities = []

    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            set1, set2 = token_sets[i], token_sets[j]
            if not set1 and not set2:
                sim = 1.0
            elif not set1 or not set2:
                sim = 0.0
            else:
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                sim = intersection / union if union > 0 else 0.0
            similarities.append(sim)

    return float(np.mean(similarities)) if similarities else 1.0
