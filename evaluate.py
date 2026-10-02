import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score


def evaluate(run_filepath, gold_filepath):
    # Load the gold standard labels
    gold_df = pd.read_csv(gold_filepath)

    # Load the model predictions
    run_df = pd.read_csv(run_filepath)

    scores = {}
    dialects = ["en-AU", "en-UK"]

    for task in ["sentiment", "sarcasm"]:
        for dialect in dialects:
            gold_filtered = gold_df[gold_df["variety"] == dialect][task]
            run_filtered = run_df[run_df["variety"] == dialect][task]

            f1 = f1_score(
                gold_filtered, run_filtered, average="macro"
            )
            print(f"F1 score for {task} ({dialect}): {f1:.4f}")
            scores[f"{task}-{dialect}"] = f1

    sent_scores = [scores[f"sentiment-{d}"] for d in dialects]
    sarc_scores = [scores[f"sarcasm-{d}"]   for d in dialects]
    final_score = (min(sent_scores) + min(sarc_scores)) / 2
    print(f"Final score: {final_score:.4f}")

    return scores, final_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against gold labels."
    )
    parser.add_argument(
        "run_filepath",
        help="Path to the model predictions CSV file.",
    )
    parser.add_argument(
        "gold_filepath",
        nargs="?",
        default="test.csv",
        help="Path to the gold standard CSV file (default: test.csv).",
    )
    parser.add_argument(
        "output_filepath",
        nargs="?",
        default="scores.txt",
        help="Path to write the scores to (default: scores.txt).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_filepath    = args.run_filepath
    gold_filepath   = args.gold_filepath
    output_filepath = args.output_filepath

    if not os.path.isfile(run_filepath):
        print(f"Run file not found: {run_filepath}")
        sys.exit(1)

    if not os.path.isfile(gold_filepath):
        print(f"Gold file not found: {gold_filepath}")
        sys.exit(1)

    scores, final_score = evaluate(run_filepath, gold_filepath)

    print("\nComponent scores:")
    for key, value in scores.items():
        print(f"f1-{key}: {value:.4f}")
    print(f"\nFinal score: {final_score:.4f}")

    with open(output_filepath, "w") as f:
        for key, value in scores.items():
            f.write(f"f1-{key}: {value:.4f}\n")
        f.write(f"score: {final_score:.4f}\n")

    print(f"\nWrote {output_filepath}")