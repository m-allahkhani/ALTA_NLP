import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def calculate_metrics(
    labels,
    predictions
):

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro"
        ),

        "positive_f1": f1_score(
            labels,
            predictions,
            pos_label=1
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        )
    }