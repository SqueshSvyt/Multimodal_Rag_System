import json
from collections import Counter
from typing import List, Dict, Any

import numpy as np


def evaluate_binary_output(outputs: List[str]) -> dict:
    """Evaluate a list of binary outputs (1 for correct, 0 for incorrect).

    Args:
        outputs (List[str]): List of binary strings ("0" or "1").

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Convert strings to integers
    outputs = [int(x) for x in outputs]

    # Count total, correct, and incorrect
    total = len(outputs)
    correct_count = sum(outputs)
    incorrect_count = total - correct_count

    # Calculate accuracy
    accuracy = correct_count / total if total > 0 else 0.0
    correct_percentage = accuracy * 100
    incorrect_percentage = (incorrect_count / total) * 100 if total > 0 else 0.0

    # Calculate class imbalance ratio (correct to incorrect)
    imbalance_ratio = correct_count / incorrect_count if incorrect_count > 0 else float('inf')

    return {
        "total": total,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "accuracy": accuracy,
        "correct_percentage": correct_percentage,
        "incorrect_percentage": incorrect_percentage,
        "imbalance_ratio": imbalance_ratio
    }


def evaluate_metrics(evaluations: List[Dict[str, int]]) -> Dict[str, Any]:
    """Evaluate a list of response evaluations with multiple metrics.

    Computes aggregate statistics (mean, median, std, min, max) for each metric
    (faithfulness, relevance, groundedness, helpfulness) and an overall average score.

    Args:
        evaluations (List[Dict[str, int]]): List of dictionaries, each containing
            scores for 'faithfulness', 'relevance', 'groundedness', and 'helpfulness'.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - Per-metric statistics (mean, median, std, min, max, distribution).
            - Overall average score across all metrics and responses.
            - Total number of responses.
    """
    # Validate input
    if not evaluations:
        return {"error": "Empty evaluation list", "metrics": {}, "overall_average": 0.0, "total_responses": 0}

    metrics = ["faithfulness", "relevance", "groundedness", "helpfulness"]
    result = {"metrics": {}, "total_responses": len(evaluations)}

    # Collect scores for each metric
    scores = {metric: [] for metric in metrics}
    all_scores = []  # For overall average

    for eval_dict in evaluations:
        for metric in metrics:
            if metric in eval_dict:
                score = eval_dict[metric]
                scores[metric].append(score)
                all_scores.append(score)

    # Compute statistics for each metric
    for metric in metrics:
        metric_scores = np.array(scores[metric])
        result["metrics"][metric] = {
            "mean": np.mean(metric_scores),
            "median": np.median(metric_scores),
            "std": np.std(metric_scores),
            "min": np.min(metric_scores),
            "max": np.max(metric_scores),
            "distribution": dict(Counter(metric_scores))
        }

    # Compute overall average score
    result["overall_average"] = np.mean(all_scores) if all_scores else 0.0

    return result


def load_json_file(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Evaluate metrics
metrics_result = evaluate_metrics(load_json_file('0_5_model_answer.json'))
print("Accurancy answer:", evaluate_binary_output(load_json_file("0_1_answer.json")))

# Print results
print("Evaluation Metrics:")
print(f"Total Responses: {metrics_result['total_responses']}")
print(f"Overall Average Score: {metrics_result['overall_average']:.2f}")
for metric, stats in metrics_result["metrics"].items():
    print(f"\n{metric.capitalize()}:")
    print(f"  Mean: {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std: {stats['std']:.2f}")
    print(f"  Min: {stats['min']}")
    print(f"  Max: {stats['max']}")
    print(f"  Distribution: {stats['distribution']}")
