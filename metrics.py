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


def evaluate_final_test(
    varieties,
    sentiment_labels,
    sentiment_predictions,
    sarcasm_labels,
    sarcasm_predictions
):
   

    varieties = np.asarray(varieties)

    sentiment_labels = np.asarray(
        sentiment_labels
    )

    sentiment_predictions = np.asarray(
        sentiment_predictions
    )

    sarcasm_labels = np.asarray(
        sarcasm_labels
    )

    sarcasm_predictions = np.asarray(
        sarcasm_predictions
    )


    n = len(varieties)

    if not (
        len(sentiment_labels) == n
        and
        len(sentiment_predictions) == n
        and
        len(sarcasm_labels) == n
        and
        len(sarcasm_predictions) == n
    ):
        raise ValueError(
            "All input arrays must have the same length."
        )


    au_mask = (
        varieties == "en-AU"
    )

    uk_mask = (
        varieties == "en-UK"
    )

    if not np.any(au_mask):
        raise ValueError(
            "No en-AU samples found."
        )

    if not np.any(uk_mask):
        raise ValueError(
            "No en-UK samples found."
        )

  
    sentiment_en_au = f1_score(
        sentiment_labels[au_mask],
        sentiment_predictions[au_mask],
        average="macro"
    )

    sentiment_en_uk = f1_score(
        sentiment_labels[uk_mask],
        sentiment_predictions[uk_mask],
        average="macro"
    )

    sarcasm_en_au = f1_score(
        sarcasm_labels[au_mask],
        sarcasm_predictions[au_mask],
        average="macro"
    )

    sarcasm_en_uk = f1_score(
        sarcasm_labels[uk_mask],
        sarcasm_predictions[uk_mask],
        average="macro"
    )


    sentiment_score = min(
        sentiment_en_au,
        sentiment_en_uk
    )

    sarcasm_score = min(
        sarcasm_en_au,
        sarcasm_en_uk
    )

    final_score = (
        sentiment_score
        +
        sarcasm_score
    ) / 2.0

    return {

        "f1-sentiment-en-AU":
            sentiment_en_au,

        "f1-sentiment-en-UK":
            sentiment_en_uk,

        "f1-sarcasm-en-AU":
            sarcasm_en_au,

        "f1-sarcasm-en-UK":
            sarcasm_en_uk,

        "score":
            final_score
    }