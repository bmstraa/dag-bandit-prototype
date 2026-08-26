# src/dag_engine.py
import torch
from collections import Counter
from .verifier import verify_with_rules, evaluate_answer, calculate_jaccard_consistency

# Mock knowledge base (the "Tool")
MOCK_KNOWLEDGE_BASE = {
    "what is the chemical symbol for water?": "H2O",
    "what is the speed of light in kilometers per second?": "299792",
    "what is the capital of brazil?": "Brasilia"
}

def run_dag(
    question: str,
    ground_truth: str,
    model,
    tokenizer,
    threshold: float = 0.8,
    mock_kb: dict = None
) -> dict:
    """
    Executes the 3-node DAG: Generator -> Semantic Verifier -> Factual Verifier -> (Optional) Tool Fallback.
    
    Args:
        question: The user query.
        ground_truth: The correct answer (for evaluation).
        model: The loaded HuggingFace model.
        tokenizer: The corresponding tokenizer.
        threshold: Decision gate threshold (0.8 by default).
        mock_kb: Optional dictionary to use as the tool. Defaults to MOCK_KNOWLEDGE_BASE.
    
    Returns:
        A dictionary with keys: question, generated_answers, semantic_conf, factual_conf,
        final_answer, tool_used, correct.
    """
    if mock_kb is None:
        mock_kb = MOCK_KNOWLEDGE_BASE

    # 1. Generator: sample 3 responses
    answers = []
    for _ in range(3):
        prompt = f"Answer this concisely: {question}"
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_ids = outputs[:, inputs.input_ids.shape[1]:]
        ans = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        answers.append(ans)

    # 2. Semantic Verifier (Jaccard consistency)
    semantic_conf = calculate_jaccard_consistency(answers)

    # 3. Factual Verifier (rule-based check on the most common answer)
    most_common_answer = Counter(answers).most_common(1)[0][0]
    factual_correct, factual_conf = verify_with_rules(most_common_answer, ground_truth)

    # 4. Decision Gate
    if factual_conf < threshold or semantic_conf < threshold:
        tool_result = mock_kb.get(question.lower(), "Tool: No info found.")
        final_answer = f"{tool_result} (via Tool)"
        tool_used = True
    else:
        final_answer = most_common_answer
        tool_used = False

    # 5. Final Evaluation
    correct = evaluate_answer(final_answer, ground_truth)

    return {
        "question": question,
        "generated_answers": answers,
        "semantic_conf": semantic_conf,
        "factual_conf": factual_conf,
        "final_answer": final_answer,
        "tool_used": tool_used,
        "correct": correct
    }
