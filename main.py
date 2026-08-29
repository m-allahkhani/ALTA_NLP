import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from sklearn.model_selection import (
    StratifiedKFold
)

from dataset import (
    ALTAMultiTaskDataset
)

from model import (
    DialectAwareMultiTaskDeBERTa
)

from metrics import (
    calculate_metrics
)
from focal_loss import FocalLoss
from multi_task_loss import MultiTaskLoss

# ==================================================
# Configuration
# ==================================================

SEED = 42

MODEL_NAME = (
    "microsoft/deberta-v3-base"
)

# MAX_LENGTH = 256
MAX_LENGTH = 128

# BATCH_SIZE = 8
BATCH_SIZE = 2

# GRADIENT_ACCUMULATION_STEPS = 2
GRADIENT_ACCUMULATION_STEPS = 4

NUM_EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

# NUM_FOLDS = 5

NUM_FOLDS = 2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# Reproducibility
# ==================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(
        seed
    )

    torch.backends.cudnn.deterministic = (
        True
    )

    torch.backends.cudnn.benchmark = (
        False
    )


set_seed(SEED)





# ==================================================
# Class weights
# ==================================================

def compute_class_weights(labels):

    labels = np.asarray(
        labels
    )

    counts = np.bincount(
        labels,
        minlength=2
    )

    weights = (
        len(labels)
        /
        (2 * counts)
    )

    return torch.tensor(
        weights,
        dtype=torch.float
    )


# ==================================================
# Training
# ==================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_function
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(
        dataloader
    ):

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        variety_ids = batch[
            "variety_ids"
        ].to(DEVICE)

        sentiment_labels = batch[
            "sentiment_labels"
        ].to(DEVICE)

        sarcasm_labels = batch[
            "sarcasm_labels"
        ].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            variety_ids=variety_ids
        )

        losses = loss_function(
            sentiment_logits=
                outputs["sentiment_logits"],

            sentiment_labels=
                sentiment_labels,

            sarcasm_logits=
                outputs["sarcasm_logits"],

            sarcasm_labels=
                sarcasm_labels
        )

        loss = (
            losses["loss"]
            /
            GRADIENT_ACCUMULATION_STEPS
        )

        loss.backward()

        if (
            (step + 1)
            %
            GRADIENT_ACCUMULATION_STEPS
            ==
            0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

        total_loss += (
            loss.item()
            *
            GRADIENT_ACCUMULATION_STEPS
        )

    return (
        total_loss
        /
        len(dataloader)
    )


# ==================================================
# Validation
# ==================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    loss_function
):

    model.eval()

    total_loss = 0.0

    sentiment_labels_all = []
    sentiment_predictions_all = []

    sarcasm_labels_all = []
    sarcasm_predictions_all = []

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        variety_ids = batch[
            "variety_ids"
        ].to(DEVICE)

        sentiment_labels = batch[
            "sentiment_labels"
        ].to(DEVICE)

        sarcasm_labels = batch[
            "sarcasm_labels"
        ].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            variety_ids=variety_ids
        )

        losses = loss_function(
            sentiment_logits=
                outputs["sentiment_logits"],

            sentiment_labels=
                sentiment_labels,

            sarcasm_logits=
                outputs["sarcasm_logits"],

            sarcasm_labels=
                sarcasm_labels
        )

        total_loss += (
            losses["loss"].item()
        )

        sentiment_predictions = (
            torch.argmax(
                outputs[
                    "sentiment_logits"
                ],
                dim=1
            )
        )

        sarcasm_predictions = (
            torch.argmax(
                outputs[
                    "sarcasm_logits"
                ],
                dim=1
            )
        )

        sentiment_labels_all.extend(
            sentiment_labels.cpu().numpy()
        )

        sentiment_predictions_all.extend(
            sentiment_predictions.cpu().numpy()
        )

        sarcasm_labels_all.extend(
            sarcasm_labels.cpu().numpy()
        )

        sarcasm_predictions_all.extend(
            sarcasm_predictions.cpu().numpy()
        )

    sentiment_metrics = (
        calculate_metrics(
            sentiment_labels_all,
            sentiment_predictions_all
        )
    )

    sarcasm_metrics = (
        calculate_metrics(
            sarcasm_labels_all,
            sarcasm_predictions_all
        )
    )

    return {

        "loss": (
            total_loss
            /
            len(dataloader)
        ),

        "sentiment": sentiment_metrics,

        "sarcasm": sarcasm_metrics
    }



def main():

    CSV_PATH = f"E:\\PROJECTS\\Alta2026\\Project\\data\\official_data\\train.csv" 
    df = pd.read_csv(
       CSV_PATH
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # ----------------------------------------------
    # Stratification
    # ----------------------------------------------

    df["stratify_group"] = (
        df["variety"].astype(str)
        +
        "_"
        +
        df["sarcasm"].astype(str)
    )

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    fold_scores = []

    for fold, (
        train_indices,
        val_indices
    ) in enumerate(
        skf.split(
            df,
            df["stratify_group"]
        )
    ):

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"FOLD {fold + 1}"
        )

        print(
            f"{'=' * 60}"
        )

        train_df = df.iloc[
            train_indices
        ].copy()

        val_df = df.iloc[
            val_indices
        ].copy()

        # ------------------------------------------
        # Datasets
        # ------------------------------------------

        train_dataset = (
            ALTAMultiTaskDataset(
                dataframe=train_df,
                tokenizer=tokenizer,
                max_length=MAX_LENGTH
            )
        )

        val_dataset = (
            ALTAMultiTaskDataset(
                dataframe=val_df,
                tokenizer=tokenizer,
                max_length=MAX_LENGTH
            )
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        # ------------------------------------------
        # Model
        # ------------------------------------------

        model = (
            DialectAwareMultiTaskDeBERTa(
                model_name=MODEL_NAME,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                dropout=0.1
            )
        )

        model.to(
            DEVICE
        )

        # ------------------------------------------
        # Losses
        # ------------------------------------------

        sentiment_weights = (
            compute_class_weights(
                train_df[
                    "sentiment"
                ]
            )
            .to(DEVICE)
        )

        sarcasm_weights = (
            compute_class_weights(
                train_df[
                    "sarcasm"
                ]
            )
            .to(DEVICE)
        )

        sentiment_loss = (
            nn.CrossEntropyLoss(
                weight=sentiment_weights
            )
        )

        sarcasm_loss = (
            FocalLoss(
                alpha=sarcasm_weights,
                gamma=2.0
            )
        )

        loss_function = (
            MultiTaskLoss(
                sentiment_loss=sentiment_loss,
                sarcasm_loss=sarcasm_loss,
                sentiment_weight=0.4,
                sarcasm_weight=0.6
            )
        )

        # ------------------------------------------
        # Optimizer
        # ------------------------------------------

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        # ------------------------------------------
        # Scheduler
        # ------------------------------------------

        total_training_steps = (
            len(train_loader)
            *
            NUM_EPOCHS
            //
            GRADIENT_ACCUMULATION_STEPS
        )

        warmup_steps = int(
            total_training_steps
            *
            WARMUP_RATIO
        )

        scheduler = (
            get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=
                    warmup_steps,

                num_training_steps=
                    total_training_steps
            )
        )

        # ------------------------------------------
        # Train
        # ------------------------------------------

        best_score = -1.0

        for epoch in range(
            NUM_EPOCHS
        ):

            print(
                f"\nEpoch "
                f"{epoch + 1}/"
                f"{NUM_EPOCHS}"
            )

            train_loss = (
                train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_function=
                        loss_function
                )
            )

            results = evaluate(
                model=model,
                dataloader=val_loader,
                loss_function=
                    loss_function
            )

            print(
                f"Train Loss: "
                f"{train_loss:.4f}"
            )

            print(
                f"Validation Loss: "
                f"{results['loss']:.4f}"
            )

            print(
                "Sentiment "
                f"Macro F1: "
                f"{results['sentiment']['macro_f1']:.4f}"
            )

            print(
                "Sarcasm "
                f"Macro F1: "
                f"{results['sarcasm']['macro_f1']:.4f}"
            )

            # --------------------------------------
            # Combined model selection score
            # --------------------------------------

            combined_score = (
                0.5
                *
                results[
                    "sentiment"
                ][
                    "macro_f1"
                ]

                +

                0.5
                *
                results[
                    "sarcasm"
                ][
                    "macro_f1"
                ]
            )

            if (
                combined_score
                >
                best_score
            ):

                best_score = (
                    combined_score
                )

                checkpoint_path = (
                    f"checkpoints/"
                    f"best_fold_"
                    f"{fold + 1}.pt"
                )

                torch.save(
                    {
                        "model_state_dict":
                            model.state_dict(),

                        "best_score":
                            best_score,

                        "fold":
                            fold + 1
                    },
                    checkpoint_path
                )

                print(
                    f"Saved best model "
                    f"→ {checkpoint_path}"
                )

        fold_scores.append(
            best_score
        )

        print(
            f"Best Fold Score: "
            f"{best_score:.4f}"
        )

    # ----------------------------------------------
    # Final results
    # ----------------------------------------------

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "CROSS-VALIDATION RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"Fold scores: "
        f"{fold_scores}"
    )

    print(
        f"Mean: "
        f"{np.mean(fold_scores):.4f}"
    )

    print(
        f"Std: "
        f"{np.std(fold_scores):.4f}"
    )


if __name__ == "__main__":
    main()