"""
This script performs calibration for an uncertainty detection method (e.g., EPR).

It loads model generation data, calculates uncertainty scores, and determines an
optimal threshold for classifying answers as 'high uncertainty' (likely incorrect)
based on ground truth labels.

The script outputs:
- The optimal uncertainty threshold.
- Precision, recall, and F1-score at that threshold.
- A plot visualizing the score distributions for correct and incorrect answers.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from artefactual.scoring.epr import EPR

# Add the project root to the path to allow importing from 'artefactual'
sys.path.append(str(Path(__file__).parent.parent.parent))


# The uncertainty detector expects a specific structure (RequestOutput protocol).
# We will create mock objects to adapt our JSON data to this structure.
class MockLogProbValue:
    def __init__(self, logprob):
        self.logprob = logprob


class MockCompletionOutput:
    def __init__(self, logprobs):
        self.logprobs = logprobs


class MockRequestOutput:
    def __init__(self, outputs):
        self.outputs = outputs


def load_data(file_path: str) -> list[dict]:
    """Loads generation data from a JSON file."""
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def is_correct(generated_text: str, ground_truth: str) -> bool:
    """
    Determines if the generated answer is correct.
    This is a simple check; you may want to replace it with a more robust method.
    """
    return ground_truth.lower() in generated_text.lower()


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
    """
    Finds the optimal threshold for classifying uncertain answers by maximizing F1-score.

    Args:
        scores: Array of uncertainty scores.
        labels: Array of binary labels (1 for correct, 0 for incorrect).

    Returns:
        A tuple containing (best_threshold, best_precision, best_recall, best_f1).
    """
    best_f1 = -1

    # We test thresholds from the min to max score
    thresholds = np.linspace(scores.min(), scores.max(), 100)

    for threshold in thresholds:
        # Predictions: 1 if score > threshold (high uncertainty), else 0
        predictions = (scores > threshold).astype(int)
        # True labels for this task: 1 if incorrect, 0 if correct
        true_labels = 1 - labels

        _precision, _recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average="binary", zero_division=0
        )

        best_f1 = max(best_f1, f1)


def plot_distributions(scores: np.ndarray, labels: np.ndarray, threshold: float, output_path: str):
    """
    Plots the distributions of uncertainty scores for correct and incorrect answers.
    Falls back to matplotlib.hist if seaborn is not available.
    """
    plt.figure(figsize=(12, 7))
    correct_scores = scores[labels == 1]
    incorrect_scores = scores[labels == 0]

    plt.hist(correct_scores, bins=50, density=True, color="green", alpha=0.6, label="Correct Answers")
    plt.hist(incorrect_scores, bins=50, density=True, color="red", alpha=0.6, label="Incorrect Answers")

    plt.axvline(threshold, color="blue", linestyle="--", label=f"Optimal Threshold ({threshold:.2f})")
    plt.title("Distribution of Uncertainty Scores (EPR)")
    plt.xlabel("EPR Score (Higher score = Higher uncertainty)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path)
    print(f"Distribution plot saved to {output_path}")
    plt.title("Distribution of Uncertainty Scores (EPR)")
    plt.xlabel("EPR Score (Higher score = Higher uncertainty)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(output_path)
    print(f"Distribution plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate uncertainty scores.")
    parser.add_argument(
        "--input-file",
        type=str,
        default="sample_qa_data.json",
        help="Path to the input JSON file with generation data.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="calibration_plot.png",
        help="Path to save the output plot.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Number of top log probabilities to consider for EPR calculation.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    data = load_data(args.input_file)

    print("Calculating uncertainty scores...")
    epr_detector = EPR(K=args.k)

    all_scores = []
    all_labels = []

    for item in tqdm(data, desc="Processing items"):
        # This assumes your JSON has 'logprobs', 'generated_text', and 'ground_truth_answer' fields.
        # You may need to adapt this to your actual data structure.
        if "logprobs" not in item or "ground_truth_answer" not in item or "generated_text" not in item:
            continue

        # Reconstruct the logprobs structure expected by the detector
        logprobs_data = [
            {k: MockLogProbValue(v) for k, v in token_logprobs.items()} for token_logprobs in item["logprobs"]
        ]

        mock_output = MockRequestOutput([MockCompletionOutput(logprobs_data)])

        # Compute score
        score = epr_detector.compute([mock_output])[0]
        all_scores.append(score)

        # Determine correctness
        correct = is_correct(item["generated_text"], item["ground_truth_answer"])
        all_labels.append(1 if correct else 0)

    scores_array = np.array(all_scores)
    labels_array = np.array(all_labels)

    if len(scores_array) == 0:
        print("No valid items with logprobs found in the input file. Exiting.")
        return

    print("\nFinding optimal threshold...")
    threshold, precision, recall, f1 = find_optimal_threshold(scores_array, labels_array)

    print("\n--- Calibration Results ---")
    print(f"Optimal Threshold: {threshold:.4f}")
    print("This threshold is optimized to detect INCORRECT answers.")
    print(f"  - Precision: {precision:.2%}")
    print(f"  - Recall: {recall:.2%}")
    print(f"  - F1-Score: {f1:.2%}")
    print("---------------------------\n")

    plot_distributions(scores_array, labels_array, threshold, args.output_plot)


if __name__ == "__main__":
    main()
