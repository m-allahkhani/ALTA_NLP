import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ALTAMultiTaskDataset
from model import DialectAwareMultiTaskDeBERTa





MODEL_NAME = project_main.MODEL_NAME
MAX_LENGTH = project_main.MAX_LENGTH
BATCH_SIZE = project_main.BATCH_SIZE
DEVICE = project_main.DEVICE
PREPROCESS_TEXT = project_main.preprocess_text

CHECKPOINT_DIR = getattr(
    project_main,
    "ALTA_CHECKPOINT_DIR",
    "checkpoints/task_specific",
)

if not os.path.isabs(CHECKPOINT_DIR):
    CHECKPOINT_DIR = os.path.join(
        os.getcwd(),
        CHECKPOINT_DIR,
    )

TEST_PATH = "/content/test.csv"
OUTPUT_PATH = "/content/answer.csv"

NUM_FOLDS = 3
EXPECTED_TEST_ROWS = 1573


# ============================================================
# CHECKS
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

if not os.path.isfile(TEST_PATH):
    raise FileNotFoundError(
        f"Test CSV not found:\n{TEST_PATH}"
    )


# ============================================================
# HELPERS
# ============================================================


def checkpoint_path(fold_index):
    return os.path.join(
        CHECKPOINT_DIR,
        f"best_fold_{fold_index + 1}.pt",
    )


def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} missing columns: {missing}"
        )


def make_model():
    return DialectAwareMultiTaskDeBERTa(
        model_name=MODEL_NAME,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        dropout=0.1,
    ).to(DEVICE)


def load_checkpoint(model, path, label):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        not isinstance(checkpoint, dict)
        or "model_state_dict" not in checkpoint
    ):
        raise RuntimeError(
            f"Invalid checkpoint: {path}"
        )

    missing, unexpected = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=False,
    )

    # The known checkpoints can legitimately have base-model naming
    # differences, so only reject missing/unexpected trained LoRA or heads.
    lora_missing = [
        k for k in missing
        if "lora_" in k.lower()
    ]
    lora_unexpected = [
        k for k in unexpected
        if "lora_" in k.lower()
    ]

    head_missing = [
        k for k in missing
        if (
            "sentiment_head" in k
            or "sarcasm_head" in k
        )
    ]

    head_unexpected = [
        k for k in unexpected
        if (
            "sentiment_head" in k
            or "sarcasm_head" in k
        )
    ]

    print(
        f"[{label}] "
        f"missing={len(missing)} "
        f"unexpected={len(unexpected)} "
        f"LoRA_missing={len(lora_missing)} "
        f"LoRA_unexpected={len(lora_unexpected)} "
        f"head_missing={len(head_missing)} "
        f"head_unexpected={len(head_unexpected)}"
    )

    if (
        lora_missing
        or lora_unexpected
        or head_missing
        or head_unexpected
    ):
        raise RuntimeError(
            f"{label}: trained LoRA/task-head checkpoint mismatch."
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return checkpoint


def make_loader(df, tokenizer):
    # Keep the same text preparation used by the existing task-specific
    # inference pipeline that produced your working fold checkpoints.
    model_df = df.copy()
    model_df["text"] = (
        model_df["text"]
        .fillna("")
        .astype(str)
        .apply(PREPROCESS_TEXT)
    )

    dataset = ALTAMultiTaskDataset(
        model_df,
        tokenizer,
        MAX_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


def predict_fold(model, loader):
    sentiment_probs = []
    sarcasm_probs = []

    model.eval()

    with torch.inference_mode():
        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            # IMPORTANT:
            # Native routing = use each row's real variety_id.
            # No cross-dialect swap is applied.
            variety_ids = batch[
                "variety_ids"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                variety_ids=variety_ids,
            )

            sentiment_logits = (
                output[
                    "sentiment_logits"
                ]
                .float()
            )

            sarcasm_logits = (
                output[
                    "sarcasm_logits"
                ]
                .float()
            )

            sentiment_probs.append(
                torch.softmax(
                    sentiment_logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

            sarcasm_probs.append(
                torch.softmax(
                    sarcasm_logits,
                    dim=1,
                )[:, 1].cpu().numpy()
            )

            del input_ids
            del attention_mask
            del variety_ids
            del output
            del sentiment_logits
            del sarcasm_logits

    if not sentiment_probs:
        raise RuntimeError(
            "No predictions produced."
        )

    return (
        np.concatenate(
            sentiment_probs
        ).astype(np.float64),
        np.concatenate(
            sarcasm_probs
        ).astype(np.float64),
    )


# ============================================================
# LOAD TEST
# ============================================================

print("=" * 80)
print("ALTA 2026 — BEST FOLD TEST INFERENCE ONLY")
print("=" * 80)
print("GPU:", torch.cuda.get_device_name(0))
print("Model:", MODEL_NAME)
print("Test:", TEST_PATH)
print("Checkpoints:", CHECKPOINT_DIR)
print("Folds:", NUM_FOLDS)
print("Routing: NATIVE")
print("CV / validation: NONE")
print("Cross-adaptation: NONE")
print("Threshold: 0.50")
print("=" * 80)

test_df = pd.read_csv(TEST_PATH)

require_columns(
    test_df,
    ["source", "variety", "text"],
    "test.csv",
)

if len(test_df) != EXPECTED_TEST_ROWS:
    raise RuntimeError(
        f"Expected {EXPECTED_TEST_ROWS} test rows, "
        f"found {len(test_df)}"
    )

test_df = test_df.reset_index(drop=True)

print("Test rows:", len(test_df))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DATA LOADER
# ============================================================

test_loader = make_loader(
    test_df,
    tokenizer,
)


# ============================================================
# 3 BEST-FOLD MODELS
# ============================================================
#
# One model at a time is kept on the GPU.
# This avoids unnecessary VRAM usage.
# ============================================================

sentiment_fold_probs = []
sarcasm_fold_probs = []


for fold_index in range(NUM_FOLDS):

    fold_number = fold_index + 1

    print("\n" + "-" * 80)
    print(
        f"PREDICTING WITH best_fold_{fold_number}.pt"
    )
    print("-" * 80)

    path = checkpoint_path(
        fold_index
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing checkpoint:\n{path}"
        )

    print("Checkpoint:", path)

    model = make_model()

    checkpoint = load_checkpoint(
        model,
        path,
        f"Fold {fold_number}",
    )

    sent_probs, sarc_probs = predict_fold(
        model,
        test_loader,
    )

    if len(sent_probs) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Fold {fold_number} sentiment prediction count: "
            f"{len(sent_probs)}"
        )

    if len(sarc_probs) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Fold {fold_number} sarcasm prediction count: "
            f"{len(sarc_probs)}"
        )

    sentiment_fold_probs.append(
        sent_probs
    )

    sarcasm_fold_probs.append(
        sarc_probs
    )

    print(
        f"Fold {fold_number} complete: "
        f"{len(sent_probs)} rows"
    )

    # --------------------------------------------------------
    # FREE THE ENTIRE MODEL BEFORE NEXT FOLD
    # --------------------------------------------------------

    del model
    del checkpoint
    del sent_probs
    del sarc_probs

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"✓ Fold {fold_number} GPU memory released"
    )


# ============================================================
# 3-FOLD PROBABILITY ENSEMBLE
# ============================================================

sentiment_probability = np.mean(
    np.stack(
        sentiment_fold_probs,
        axis=0,
    ),
    axis=0,
)

sarcasm_probability = np.mean(
    np.stack(
        sarcasm_fold_probs,
        axis=0,
    ),
    axis=0,
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

sentiment_predictions = (
    sentiment_probability >= 0.50
).astype(int)

sarcasm_predictions = (
    sarcasm_probability >= 0.50
).astype(int)


# ============================================================
# FINAL CHECKS
# ============================================================

if len(sentiment_predictions) != (
    EXPECTED_TEST_ROWS
):
    raise RuntimeError(
        "Final sentiment prediction count mismatch."
    )

if len(sarcasm_predictions) != (
    EXPECTED_TEST_ROWS
):
    raise RuntimeError(
        "Final sarcasm prediction count mismatch."
    )


# ============================================================
# CREATE answer.csv
# ============================================================

answer = test_df[
    [
        "source",
        "variety",
        "text",
    ]
].copy()

answer["sentiment"] = (
    sentiment_predictions
)

answer["sarcasm"] = (
    sarcasm_predictions
)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

expected_columns = [
    "source",
    "variety",
    "text",
    "sentiment",
    "sarcasm",
]

if answer.columns.tolist() != (
    expected_columns
):
    raise RuntimeError(
        "Incorrect output columns: "
        f"{answer.columns.tolist()}"
    )

if len(answer) != EXPECTED_TEST_ROWS:
    raise RuntimeError(
        f"Incorrect output row count: "
        f"{len(answer)}"
    )

if not answer[
    "sentiment"
].isin([0, 1]).all():
    raise RuntimeError(
        "Invalid sentiment labels."
    )

if not answer[
    "sarcasm"
].isin([0, 1]).all():
    raise RuntimeError(
        "Invalid sarcasm labels."
    )


# ============================================================
# SAVE answer.csv
# ============================================================

answer.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
print("Output:", OUTPUT_PATH)
print("Rows:", len(answer))
print(
    "Sentiment positive:",
    f"{answer['sentiment'].mean() * 100:.2f}%",
)
print(
    "Sarcasm positive:",
    f"{answer['sarcasm'].mean() * 100:.2f}%",
)
print("Columns:", answer.columns.tolist())
print("=" * 80)
