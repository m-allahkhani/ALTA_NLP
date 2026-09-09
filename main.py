import os
import random

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split
)

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)

from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import torch.nn.functional as F

from sklearn.model_selection import (
    StratifiedKFold
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    f1_score
)

from dataset import (
    ALTAMultiTaskDataset
)

from model import (
    DialectAwareMultiTaskDeBERTa
)

from metrics import (
    calculate_metrics,
    evaluate_final_test
)

from focal_loss import FocalLoss
from multi_task_loss import MultiTaskLoss
from general_sarcasm_pretrain_model import GeneralSarcasmPretrainModel
from rephrase_contrastive_loss import RephraseContrastiveLoss
from ISarcasm_dataset import ISarcasmDataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="ALTA Multi-Task DeBERTa Training"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Maximum number of training epochs per fold"
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=2,
        help="Number of cross-validation folds"
    )

    return parser.parse_args()

SEED = 42

# train_path = "E:\\PROJECTS\\Alta2026\\Project\\data\\official_data\\train.csv"
# test_path = "E:\\PROJECTS\\Alta2026\\Project\\data\\official_data\\valid.csv" 
# train_path = f"/kaggle//input//datasets//maryamallahkhani//official-alta-dataset//train.csv" 
# test_path = f"/kaggle//input//datasets//maryamallahkhani//official-alta-dataset//valid.csv" 
train_path = f"train.csv" 
test_path = f"valid.csv" 

MODEL_NAME = (
    "microsoft/deberta-v3-base"
)


# ============================================================
# iSarcasmEval GENERAL SARCASM PRETRAINING
# ============================================================

ISARCASM_TRAIN_PATH =  "iSarcasmEval_train.csv"


GENERAL_ADAPTER_CHECKPOINT = (
    "checkpoints/"
    "isarcasm_general_lora.pt"
)
ISARCASM_MAX_LENGTH = 128

ISARCASM_BATCH_SIZE = 4

ISARCASM_GRADIENT_ACCUMULATION_STEPS = 2

ISARCASM_EPOCHS = 8

ISARCASM_LEARNING_RATE = 1e-5

ISARCASM_WEIGHT_DECAY = 0.01

ISARCASM_WARMUP_RATIO = 0.1

ISARCASM_VAL_RATIO = 0.10

ISARCASM_PATIENCE = 2

ISARCASM_GAMMA = 2.0

ISARCASM_MIN_DELTA = 0.0005

USE_ISARCASM_PRETRAINING = True


#######################################
MAX_LENGTH = 128

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 4

# NUM_EPOCHS = 1

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

EARLY_STOPPING_PATIENCE = 3 
MIN_DELTA = 0.0005

# NUM_FOLDS = 2

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

NUM_GPUS = (
    torch.cuda.device_count()
    if torch.cuda.is_available()
    else 0
)

print("\n" + "=" * 60)
print("DEVICE CONFIGURATION")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"GPU count: {NUM_GPUS}")

if NUM_GPUS > 0:
    for i in range(NUM_GPUS):
        print(
            f"GPU {i}: "
            f"{torch.cuda.get_device_name(i)}"
        )

print("=" * 60)


# ==================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed(SEED)



def compute_class_weights(labels):

    labels = np.asarray(labels)

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

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    sentiment_loss,
    au_sarcasm_loss,
    uk_sarcasm_loss,
    sentiment_weight=0.4,
    sarcasm_weight=0.6
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        variety_ids = batch["variety_ids"].to(DEVICE)

        sentiment_labels = batch[
            "sentiment_labels"
        ].to(DEVICE)

        sarcasm_labels = batch[
            "sarcasm_labels"
        ].to(DEVICE)

        # ==================================================
        # Forward pass
        # ==================================================

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            variety_ids=variety_ids
        )

        sentiment_logits = outputs[
            "sentiment_logits"
        ]

        sarcasm_logits = outputs[
            "sarcasm_logits"
        ]

        # ==================================================
        # Sentiment loss
        # ==================================================

        sentiment_loss_value = sentiment_loss(
            sentiment_logits,
            sentiment_labels
        )

        # ==================================================
        # Dialect-specific sarcasm losses
        # ==================================================

        au_mask = (
            variety_ids == 0
        )

        uk_mask = (
            variety_ids == 1
        )

        sarcasm_loss_values = []

        # --------------------------------------------------
        # AU sarcasm
        # --------------------------------------------------

        if au_mask.any():

            au_loss = au_sarcasm_loss(
                sarcasm_logits[au_mask],
                sarcasm_labels[au_mask]
            )

            sarcasm_loss_values.append(
                au_loss
            )

        # --------------------------------------------------
        # UK sarcasm
        # --------------------------------------------------

        if uk_mask.any():

            uk_loss = uk_sarcasm_loss(
                sarcasm_logits[uk_mask],
                sarcasm_labels[uk_mask]
            )

            sarcasm_loss_values.append(
                uk_loss
            )

        # ==================================================
        # Equal dialect contribution
        # ==================================================

        if len(sarcasm_loss_values) == 2:

            sarcasm_loss_value = (
                sarcasm_loss_values[0]
                +
                sarcasm_loss_values[1]
            ) / 2.0

        elif len(sarcasm_loss_values) == 1:

            # Batch contains only one dialect.
            sarcasm_loss_value = (
                sarcasm_loss_values[0]
            )

        else:

            raise RuntimeError(
                "Batch contains neither en-AU "
                "nor en-UK samples."
            )

        # ==================================================
        # Multi-task loss
        # ==================================================

        total_batch_loss = (
            sentiment_weight *
            sentiment_loss_value
            +
            sarcasm_weight *
            sarcasm_loss_value
        )

        # ==================================================
        # Gradient accumulation
        # ==================================================

        loss = (
            total_batch_loss
            /
            GRADIENT_ACCUMULATION_STEPS
        )

        loss.backward()

        # ==================================================
        # Optimizer step
        # ==================================================

        if (
            (step + 1)
            %
            GRADIENT_ACCUMULATION_STEPS
            == 0
        ):

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

        # Track unscaled loss
        total_loss += total_batch_loss.item()

    # ==================================================
    # Handle remaining accumulated gradients
    # ==================================================

    if (
        len(dataloader)
        %
        GRADIENT_ACCUMULATION_STEPS
        != 0
    ):

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

    return (
        total_loss /
        len(dataloader)
    )

from sklearn.model_selection import train_test_split
from transformers import get_linear_schedule_with_warmup


def pretrain_general_adapter(
    train_csv = ISARCASM_TRAIN_PATH,
    model_name=MODEL_NAME,
    output_dir=GENERAL_ADAPTER_CHECKPOINT,
    max_length=ISARCASM_MAX_LENGTH,
    batch_size=ISARCASM_BATCH_SIZE,
    learning_rate=ISARCASM_LEARNING_RATE,
    weight_decay=ISARCASM_WEIGHT_DECAY,
    num_epochs=ISARCASM_EPOCHS,
    warmup_ratio=ISARCASM_WARMUP_RATIO,
    gradient_accumulation_steps=ISARCASM_GRADIENT_ACCUMULATION_STEPS,
    val_size=ISARCASM_VAL_RATIO,
    seed=42,
    num_workers=2,
):
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(train_csv)

    required_columns = [
        "tweet",
        "sarcastic",
        "rephrase",
        "sarcasm",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(f"Total samples: {len(df)}")

    print(
        "Sarcastic distribution:\n",
        df["sarcastic"].value_counts(dropna=False)
    )

    print(
        "Rephrase availability:",
        df["rephrase"].fillna("").astype(str).str.strip().ne("").sum()
    )

    print(
        "Fine-grained sarcasm distribution:\n",
        df["sarcasm"].value_counts(dropna=False)
    )

    # --------------------------------------------------
    # Stratified split using MAIN sarcastic label
    # --------------------------------------------------

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df["sarcastic"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(val_df)}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dataset = ISarcasmDataset(
        train_df,
        tokenizer,
        max_length=max_length,
    )

    val_dataset = ISarcasmDataset(
        val_df,
        tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = GeneralSarcasmPretrainModel(
        model_name=model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        dropout=0.1,
    )

    model.to(device)

    model.encoder.print_trainable_parameters()

    # --------------------------------------------------
    # Binary sarcasm class weights
    # --------------------------------------------------

    binary_counts = (
        train_df["sarcastic"]
        .value_counts()
        .sort_index()
    )

    n_negative = int(binary_counts.get(0, 1))
    n_positive = int(binary_counts.get(1, 1))

    # Standard inverse-frequency weighting
    class_weights = torch.tensor(
        [
            1.0,
            n_negative / max(n_positive, 1),
        ],
        dtype=torch.float32,
        device=device,
    )

    binary_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------
    # Fine-grained sarcasm weight
    # --------------------------------------------------

    fine_pos_weight = calculate_fine_sarcasm_pos_weight(
        train_df
    )

    print(
        f"Fine sarcasm pos_weight: "
        f"{fine_pos_weight:.4f}"
    )

    # --------------------------------------------------
    # Contrastive loss
    # --------------------------------------------------

    contrastive_loss_fn = RephraseContrastiveLoss(
        temperature=0.07
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    steps_per_epoch = max(
        1,
        (
            len(train_loader)
            + gradient_accumulation_steps
            - 1
        )
        // gradient_accumulation_steps,
    )

    total_steps = (
        steps_per_epoch * num_epochs
    )

    warmup_steps = int(
        total_steps * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):

        train_metrics = pretrain_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            binary_loss_fn=binary_loss_fn,
            fine_pos_weight=fine_pos_weight,
            contrastive_loss_fn=contrastive_loss_fn,
            binary_weight=0.60,
            rephrase_weight=0.25,
            fine_weight=0.15,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        val_metrics = evaluate_isarcasm(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch {epoch}/{num_epochs}"
        )

        print(
            f"Train Loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"  Binary: "
            f"{train_metrics['binary_loss']:.4f}"
        )

        print(
            f"  Rephrase: "
            f"{train_metrics['rephrase_loss']:.4f}"
        )

        print(
            f"  Fine sarcasm: "
            f"{train_metrics['fine_loss']:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"Validation Positive F1: "
            f"{val_metrics['positive_f1']:.4f}"
        )

        print(
            f"Validation Fine Loss: "
            f"{val_metrics['fine_loss']:.4f}"
        )

        # --------------------------------------------------
        # Save best adapter
        # --------------------------------------------------

        if val_metrics["macro_f1"] > best_f1:

            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch

            checkpoint_path = os.path.join(
                output_dir,
                "best_general_adapter.pt",
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_f1": best_f1,
                    "best_epoch": best_epoch,
                    "model_name": model_name,
                },
                checkpoint_path,
            )

            print(
                f"✓ Saved best model → "
                f"{checkpoint_path}"
            )

    print(
        f"\nBest validation Macro F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    return model, tokenizer

import os
import torch

def load_general_adapter_weights(
    model,
    checkpoint_path,
):
    """
    Load the pretrained iSarcasm general LoRA weights.

    The checkpoint path can be either:

        checkpoints/isarcasm_general_lora.pt
            -> directory containing best_general_adapter.pt

    or:

        some_checkpoint.pt
            -> direct PyTorch checkpoint file
    """

    import os
    import torch

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist:\n"
            f"{checkpoint_path}"
        )

    # ============================================================
    # If the supplied path is a directory, find the saved
    # checkpoint inside it.
    # ============================================================

    if os.path.isdir(checkpoint_path):

        candidate_paths = [
            os.path.join(
                checkpoint_path,
                "best_general_adapter.pt"
            ),
            os.path.join(
                checkpoint_path,
                "best_general_adapter.pth"
            ),
        ]

        checkpoint_file = None

        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                checkpoint_file = candidate
                break

        if checkpoint_file is None:
            raise FileNotFoundError(
                "The checkpoint path is a directory, but "
                "best_general_adapter.pt was not found.\n\n"
                f"Directory contents:\n"
                f"{os.listdir(checkpoint_path)}"
            )

    else:
        checkpoint_file = checkpoint_path

    print(
        f"Loading general adapter from:\n"
        f"{checkpoint_file}"
    )

    # ============================================================
    # Load checkpoint
    # ============================================================

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    # ============================================================
    # Extract state dictionary
    # ============================================================

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            pretrained_state = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            pretrained_state = checkpoint[
                "state_dict"
            ]

        else:
            # Assume checkpoint itself is the state_dict
            pretrained_state = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    # ============================================================
    # Current ALTA model state
    # ============================================================

    current_state = model.state_dict()

    transferred = 0
    skipped = 0

    transferred_keys = []
    skipped_keys = []

    # ============================================================
    # Transfer only LoRA encoder weights
    # ============================================================

    for source_key, source_value in pretrained_state.items():

        # We only want LoRA parameters.
        if (
            "lora_A" not in source_key
            and "lora_B" not in source_key
        ):
            continue

        matched_key = None

        # --------------------------------------------------------
        # Exact match
        # --------------------------------------------------------

        if source_key in current_state:
            matched_key = source_key

        # --------------------------------------------------------
        # default -> general adapter name
        # --------------------------------------------------------

        if matched_key is None:

            candidates = [
                source_key.replace(
                    ".lora_A.default.",
                    ".lora_A.general."
                ),
                source_key.replace(
                    ".lora_B.default.",
                    ".lora_B.general."
                ),
            ]

            for candidate in candidates:

                if candidate in current_state:
                    matched_key = candidate
                    break

        # --------------------------------------------------------
        # More flexible matching.
        #
        # We compare the layer/path while ignoring the adapter
        # name.
        # --------------------------------------------------------

        if matched_key is None:

            source_normalized = source_key

            source_normalized = (
                source_normalized
                .replace(
                    ".lora_A.default.",
                    ".lora_A.<ADAPTER>."
                )
                .replace(
                    ".lora_B.default.",
                    ".lora_B.<ADAPTER>."
                )
                .replace(
                    ".lora_A.general.",
                    ".lora_A.<ADAPTER>."
                )
                .replace(
                    ".lora_B.general.",
                    ".lora_B.<ADAPTER>."
                )
            )

            source_normalized = (
                source_normalized
                .replace(
                    "base_model.model.",
                    ""
                )
                .replace(
                    "base_model.",
                    ""
                )
            )

            for target_key in current_state.keys():

                if (
                    "lora_A" not in target_key
                    and "lora_B" not in target_key
                ):
                    continue

                target_normalized = target_key

                target_normalized = (
                    target_normalized
                    .replace(
                        ".lora_A.default.",
                        ".lora_A.<ADAPTER>."
                    )
                    .replace(
                        ".lora_B.default.",
                        ".lora_B.<ADAPTER>."
                    )
                    .replace(
                        ".lora_A.general.",
                        ".lora_A.<ADAPTER>."
                    )
                    .replace(
                        ".lora_B.general.",
                        ".lora_B.<ADAPTER>."
                    )
                )

                target_normalized = (
                    target_normalized
                    .replace(
                        "base_model.model.",
                        ""
                    )
                    .replace(
                        "base_model.",
                        ""
                    )
                )

                if (
                    source_normalized
                    == target_normalized
                    and current_state[target_key].shape
                    == source_value.shape
                ):
                    matched_key = target_key
                    break

        # --------------------------------------------------------
        # Transfer
        # --------------------------------------------------------

        if matched_key is not None:

            current_state[matched_key] = (
                source_value.to(
                    device=current_state[
                        matched_key
                    ].device,
                    dtype=current_state[
                        matched_key
                    ].dtype,
                )
            )

            transferred += 1
            transferred_keys.append(
                (source_key, matched_key)
            )

        else:
            skipped += 1
            skipped_keys.append(source_key)

    # ============================================================
    # Load into model
    # ============================================================

    model.load_state_dict(
        current_state,
        strict=False,
    )

    # ============================================================
    # Report
    # ============================================================

    print()
    print("=" * 60)
    print("GENERAL LoRA TRANSFER")
    print("=" * 60)

    print(
        f"Source checkpoint : {checkpoint_file}"
    )

    print(
        f"Transferred       : {transferred}"
    )

    print(
        f"Skipped            : {skipped}"
    )

    print("=" * 60)

    if transferred == 0:
        raise RuntimeError(
            "No LoRA weights were transferred.\n\n"
            "This means the parameter names in the iSarcasm "
            "checkpoint do not match the LoRA parameters in "
            "the ALTA model."
        )

    print(
        "✓ General sarcasm LoRA loaded successfully."
    )

    return model
from sklearn.metrics import f1_score


@torch.no_grad()
def evaluate_isarcasm(
    model,
    loader,
    device,
):
    model.eval()

    all_targets = []
    all_predictions = []

    total_fine_loss = 0.0
    fine_count = 0

    for batch in loader:
        tweet_input_ids = batch[
            "tweet_input_ids"
        ].to(device)

        tweet_attention_mask = batch[
            "tweet_attention_mask"
        ].to(device)

        labels = batch[
            "sarcastic_label"
        ].to(device)

        fine_labels = batch[
            "fine_sarcasm_label"
        ].to(device)

        outputs = model(
            tweet_input_ids=tweet_input_ids,
            tweet_attention_mask=tweet_attention_mask,
        )

        predictions = outputs[
            "binary_logits"
        ].argmax(dim=-1)

        all_targets.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        # Fine category metric
        valid = fine_labels >= 0

        if valid.any():
            fine_logits = outputs[
                "fine_logits"
            ][valid]

            fine_targets = fine_labels[
                valid
            ].float()

            fine_loss = F.binary_cross_entropy_with_logits(
                fine_logits,
                fine_targets,
            )

            total_fine_loss += (
                fine_loss.item()
                * int(valid.sum())
            )

            fine_count += int(valid.sum())

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    positive_f1 = f1_score(
        all_targets,
        all_predictions,
        average="binary",
        zero_division=0,
    )

    fine_loss_avg = (
        total_fine_loss / fine_count
        if fine_count > 0
        else 0.0
    )

    return {
        "macro_f1": macro_f1,
        "positive_f1": positive_f1,
        "fine_loss": fine_loss_avg,
    }


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    sentiment_loss,
    au_sarcasm_loss,
    uk_sarcasm_loss,
    sentiment_weight=0.4,
    sarcasm_weight=0.6
):
    model.eval()

    total_loss = 0.0

    # --------------------------------------------------
    # Store predictions separately for each dialect
    # --------------------------------------------------

    sentiment_labels_au = []
    sentiment_predictions_au = []

    sentiment_labels_uk = []
    sentiment_predictions_uk = []

    sarcasm_labels_au = []
    sarcasm_predictions_au = []

    sarcasm_labels_uk = []
    sarcasm_predictions_uk = []

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------

    for batch in dataloader:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        variety_ids = batch["variety_ids"].to(DEVICE)

        sentiment_labels = batch[
            "sentiment_labels"
        ].to(DEVICE)

        sarcasm_labels = batch[
            "sarcasm_labels"
        ].to(DEVICE)

        # ----------------------------------------------
        # Forward pass
        # ----------------------------------------------

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            variety_ids=variety_ids
        )

        sentiment_logits = outputs[
            "sentiment_logits"
        ]

        sarcasm_logits = outputs[
            "sarcasm_logits"
        ]

        # ==============================================
        # Sentiment loss
        # ==============================================

        sentiment_loss_value = sentiment_loss(
            sentiment_logits,
            sentiment_labels
        )

        # ==============================================
        # Dialect-specific sarcasm losses
        # ==============================================

        au_mask = (
            variety_ids == 0
        )

        uk_mask = (
            variety_ids == 1
        )

        sarcasm_loss_values = []

        # --------------------------------------------------
        # en-AU sarcasm loss
        # --------------------------------------------------

        if au_mask.any():

            au_loss = au_sarcasm_loss(
                sarcasm_logits[au_mask],
                sarcasm_labels[au_mask]
            )

            sarcasm_loss_values.append(
                au_loss
            )

        # --------------------------------------------------
        # en-UK sarcasm loss
        # --------------------------------------------------

        if uk_mask.any():

            uk_loss = uk_sarcasm_loss(
                sarcasm_logits[uk_mask],
                sarcasm_labels[uk_mask]
            )

            sarcasm_loss_values.append(
                uk_loss
            )

        # --------------------------------------------------
        # Equal AU / UK contribution
        # --------------------------------------------------

        if len(sarcasm_loss_values) == 2:

            sarcasm_loss_value = (
                sarcasm_loss_values[0]
                +
                sarcasm_loss_values[1]
            ) / 2.0

        elif len(sarcasm_loss_values) == 1:

            sarcasm_loss_value = (
                sarcasm_loss_values[0]
            )

        else:

            raise RuntimeError(
                "Batch contains neither en-AU "
                "nor en-UK samples."
            )

        # ==============================================
        # Multi-task validation loss
        # ==============================================

        total_batch_loss = (
            sentiment_weight *
            sentiment_loss_value
            +
            sarcasm_weight *
            sarcasm_loss_value
        )

        total_loss += (
            total_batch_loss.item()
        )

        # ==============================================
        # Predictions
        # ==============================================

        sentiment_predictions = torch.argmax(
            sentiment_logits,
            dim=1
        )

        sarcasm_predictions = torch.argmax(
            sarcasm_logits,
            dim=1
        )

        # Move to CPU / NumPy
        variety_ids_cpu = (
            variety_ids.cpu().numpy()
        )

        sentiment_labels_cpu = (
            sentiment_labels.cpu().numpy()
        )

        sentiment_predictions_cpu = (
            sentiment_predictions.cpu().numpy()
        )

        sarcasm_labels_cpu = (
            sarcasm_labels.cpu().numpy()
        )

        sarcasm_predictions_cpu = (
            sarcasm_predictions.cpu().numpy()
        )

        # ==============================================
        # Separate AU / UK predictions
        # ==============================================

        for i in range(
            len(variety_ids_cpu)
        ):

            variety_id = variety_ids_cpu[i]

            # ------------------------------------------
            # en-AU
            # ------------------------------------------

            if variety_id == 0:

                sentiment_labels_au.append(
                    sentiment_labels_cpu[i]
                )

                sentiment_predictions_au.append(
                    sentiment_predictions_cpu[i]
                )

                sarcasm_labels_au.append(
                    sarcasm_labels_cpu[i]
                )

                sarcasm_predictions_au.append(
                    sarcasm_predictions_cpu[i]
                )

            # ------------------------------------------
            # en-UK
            # ------------------------------------------

            elif variety_id == 1:

                sentiment_labels_uk.append(
                    sentiment_labels_cpu[i]
                )

                sentiment_predictions_uk.append(
                    sentiment_predictions_cpu[i]
                )

                sarcasm_labels_uk.append(
                    sarcasm_labels_cpu[i]
                )

                sarcasm_predictions_uk.append(
                    sarcasm_predictions_cpu[i]
                )

    # ==================================================
    # Calculate dialect-specific Macro F1
    # ==================================================

    sentiment_au_metrics = calculate_metrics(
        sentiment_labels_au,
        sentiment_predictions_au
    )

    sentiment_uk_metrics = calculate_metrics(
        sentiment_labels_uk,
        sentiment_predictions_uk
    )

    sarcasm_au_metrics = calculate_metrics(
        sarcasm_labels_au,
        sarcasm_predictions_au
    )

    sarcasm_uk_metrics = calculate_metrics(
        sarcasm_labels_uk,
        sarcasm_predictions_uk
    )

    # ==================================================
    # Extract Macro F1
    # ==================================================

    sentiment_en_au = (
        sentiment_au_metrics["macro_f1"]
    )

    sentiment_en_uk = (
        sentiment_uk_metrics["macro_f1"]
    )

    sarcasm_en_au = (
        sarcasm_au_metrics["macro_f1"]
    )

    sarcasm_en_uk = (
        sarcasm_uk_metrics["macro_f1"]
    )

    # ==================================================
    # OFFICIAL ALTA COMPETITION SCORE
    # ==================================================

    sentiment_score = min(
        sentiment_en_au,
        sentiment_en_uk
    )

    sarcasm_score = min(
        sarcasm_en_au,
        sarcasm_en_uk
    )

    official_score = (
        sentiment_score +
        sarcasm_score
    ) / 2.0

    # ==================================================
    # Return results
    # ==================================================

    return {

        "loss": (
            total_loss /
            len(dataloader)
        ),

        "sentiment": {

            "en-AU": sentiment_en_au,

            "en-UK": sentiment_en_uk,

            "macro_f1": (
                sentiment_en_au +
                sentiment_en_uk
            ) / 2.0
        },

        "sarcasm": {

            "en-AU": sarcasm_en_au,

            "en-UK": sarcasm_en_uk,

            "macro_f1": (
                sarcasm_en_au +
                sarcasm_en_uk
            ) / 2.0
        },

        "sentiment_score": sentiment_score,

        "sarcasm_score": sarcasm_score,

        "official_score": official_score
    }
@torch.no_grad()
def predict_ensemble(
    models,
    dataloader
):
    
    for model in models:
        model.eval()

    varieties = []

    sentiment_labels_all = []
    sarcasm_labels_all = []

    sentiment_logits_all_models = []
    sarcasm_logits_all_models = []

    
    for model_index, model in enumerate(models):

        sentiment_logits_model = []
        sarcasm_logits_model = []

        sentiment_labels_model = []
        sarcasm_labels_model = []

        varieties_model = []

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

            sentiment_logits_model.append(
                outputs[
                    "sentiment_logits"
                ].cpu()
            )

            sarcasm_logits_model.append(
                outputs[
                    "sarcasm_logits"
                ].cpu()
            )

            sentiment_labels_model.extend(
                sentiment_labels.cpu().numpy()
            )

            sarcasm_labels_model.extend(
                sarcasm_labels.cpu().numpy()
            )

            varieties_model.extend(
                batch["variety"]
            )

        sentiment_logits_model = torch.cat(
            sentiment_logits_model,
            dim=0
        )

        sarcasm_logits_model = torch.cat(
            sarcasm_logits_model,
            dim=0
        )

        sentiment_logits_all_models.append(
            sentiment_logits_model
        )

        sarcasm_logits_all_models.append(
            sarcasm_logits_model
        )

        if model_index == 0:

            sentiment_labels_all = (
                sentiment_labels_model
            )

            sarcasm_labels_all = (
                sarcasm_labels_model
            )

            varieties = varieties_model

 

    sentiment_logits_ensemble = torch.stack(
        sentiment_logits_all_models,
        dim=0
    ).mean(dim=0)

    sarcasm_logits_ensemble = torch.stack(
        sarcasm_logits_all_models,
        dim=0
    ).mean(dim=0)


    sentiment_predictions = torch.argmax(
        sentiment_logits_ensemble,
        dim=1
    ).numpy()

    sarcasm_predictions = torch.argmax(
        sarcasm_logits_ensemble,
        dim=1
    ).numpy()

    return {
        "varieties": varieties,

        "sentiment_labels":
            sentiment_labels_all,

        "sentiment_predictions":
            sentiment_predictions,

        "sarcasm_labels":
            sarcasm_labels_all,

        "sarcasm_predictions":
            sarcasm_predictions
    }



import re
import unicodedata


def preprocess_text(text):

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.strip()

    return text

def masked_binary_loss(
    logits,
    targets,
    pos_weight=None,
):
    """
    BCE loss while ignoring targets == -1.
    """

    valid_mask = targets >= 0

    if not valid_mask.any():
        return torch.tensor(
            0.0,
            device=logits.device,
            requires_grad=True,
        )

    valid_logits = logits[valid_mask]
    valid_targets = targets[valid_mask].float()

    return F.binary_cross_entropy_with_logits(
        valid_logits,
        valid_targets,
        pos_weight=pos_weight,
    )
def calculate_fine_sarcasm_pos_weight(df):
    values = pd.to_numeric(
        df["sarcasm"],
        errors="coerce",
    )

    values = values.dropna().astype(int)

    positive = int((values == 1).sum())
    negative = int((values == 0).sum())

    if positive == 0:
        return 1.0

    return negative / positive
def pretrain_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    binary_loss_fn,
    fine_pos_weight,
    contrastive_loss_fn,
    binary_weight=0.60,
    rephrase_weight=0.25,
    fine_weight=0.15,
    gradient_accumulation_steps=4,
):
    model.train()

    total_loss = 0.0
    total_binary = 0.0
    total_rephrase = 0.0
    total_fine = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        tweet_input_ids = batch[
            "tweet_input_ids"
        ].to(device)

        tweet_attention_mask = batch[
            "tweet_attention_mask"
        ].to(device)

        rephrase_input_ids = batch[
            "rephrase_input_ids"
        ].to(device)

        rephrase_attention_mask = batch[
            "rephrase_attention_mask"
        ].to(device)

        sarcastic_label = batch[
            "sarcastic_label"
        ].to(device)

        fine_label = batch[
            "fine_sarcasm_label"
        ].to(device)

        has_rephrase = batch[
            "has_rephrase"
        ].to(device)

        outputs = model(
            tweet_input_ids=tweet_input_ids,
            tweet_attention_mask=tweet_attention_mask,
            rephrase_input_ids=rephrase_input_ids,
            rephrase_attention_mask=rephrase_attention_mask,
        )

        # --------------------------------------------------
        # 1. Binary sarcasm loss
        # --------------------------------------------------

        loss_binary = binary_loss_fn(
            outputs["binary_logits"],
            sarcastic_label,
        )

        # --------------------------------------------------
        # 2. Rephrase contrastive loss
        # --------------------------------------------------

        valid_rephrase = has_rephrase.bool()

        if valid_rephrase.sum() >= 2:
            tweet_proj = outputs["tweet_proj"][
                valid_rephrase
            ]

            rephrase_proj = outputs["rephrase_proj"][
                valid_rephrase
            ]

            loss_rephrase = contrastive_loss_fn(
                tweet_proj,
                rephrase_proj,
            )
        else:
            loss_rephrase = torch.tensor(
                0.0,
                device=device,
                requires_grad=True,
            )

        # --------------------------------------------------
        # 3. Fine-grained sarcasm-category loss
        # --------------------------------------------------

        pos_weight_tensor = torch.tensor(
            fine_pos_weight,
            dtype=torch.float32,
            device=device,
        )

        loss_fine = masked_binary_loss(
            outputs["fine_logits"],
            fine_label,
            pos_weight=pos_weight_tensor,
        )

        # --------------------------------------------------
        # Combined loss
        # --------------------------------------------------

        loss = (
            binary_weight * loss_binary
            + rephrase_weight * loss_rephrase
            + fine_weight * loss_fine
        )

        loss_for_backward = (
            loss / gradient_accumulation_steps
        )

        loss_for_backward.backward()

        if (
            (step + 1) % gradient_accumulation_steps == 0
            or (step + 1) == len(loader)
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_binary += loss_binary.item()
        total_rephrase += loss_rephrase.item()
        total_fine += loss_fine.item()

    n = len(loader)

    return {
        "loss": total_loss / n,
        "binary_loss": total_binary / n,
        "rephrase_loss": total_rephrase / n,
        "fine_loss": total_fine / n,
    }
def print_lora_adapter_norms(model):
    print("\n" + "=" * 70)
    print("LoRA ADAPTER NORMS")
    print("=" * 70)

    for name, param in model.named_parameters():

        if "lora_A" not in name and "lora_B" not in name:
            continue

        if not param.requires_grad:
            continue

        print(
            f"{name:<100} "
            f"norm={param.detach().float().norm().item():.8f}"
        )

    print("=" * 70)
def print_all_lora_adapter_norms(model):
    print("\n" + "=" * 80)
    print("ALL LoRA ADAPTER NORMS")
    print("=" * 80)

    for name, param in model.named_parameters():

        if (
            "lora_A" not in name
            and "lora_B" not in name
        ):
            continue

        if not param.requires_grad:
            continue

        print(
            f"{name:<110} "
            f"norm={param.detach().float().norm().item():.8f}"
        )

    print("=" * 80)

def main(args):

    NUM_EPOCHS = args.epochs
    NUM_FOLDS = args.folds

    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Folds:  {NUM_FOLDS}")
    print("=" * 60)


    df = pd.read_csv(train_path)

    print( f"Training samples: {len(df)}")

    # ==================================================
    # Light text preprocessing
    # ==================================================

    original_text = df["text"].astype(str).copy()

    # Apply preprocessing
    df["text"] = (
        df["text"]
        .apply(preprocess_text)
    )

    # Compare original vs preprocessed text
    changed_mask = (
        original_text != df["text"]
    )

    changed_count = changed_mask.sum()

    unchanged_count = (
        len(df) - changed_count
    )

    changed_percentage = (
        changed_count / len(df) * 100
    )

    print("\n"+"=" * 60)

    print("PREPROCESSING SUMMARY")

    print("=" * 60)
    print(f"Total samples:     {len(df)}")

    print(f"Changed samples:   {changed_count}")

    print(f"Unchanged samples: {unchanged_count}")

    print(f"Changed percentage: {changed_percentage:.2f}%")

  

    print(f"Training samples: {len(df)}")

    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # ==================================================
    # iSarcasmEval GENERAL ADAPTER PRETRAINING
    # ==================================================

    if USE_ISARCASM_PRETRAINING:

        if os.path.exists(
            GENERAL_ADAPTER_CHECKPOINT
        ):

            print(
                "\nExisting general adapter checkpoint found:"
            )

            print(
                GENERAL_ADAPTER_CHECKPOINT
            )

            print(
                "Skipping iSarcasm pretraining."
            )

        else:

            pretrain_general_adapter()

   

    # df["stratify_group"] =  df["variety"].astype(str)+ "_" + df["sarcasm"].astype(str)
    df["stratify_group"] = df["variety"].astype(str) + "_" + df["sentiment"].astype(str)+ "_" + df["sarcasm"].astype(str)

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

  
    for fold, (train_indices,val_indices) in enumerate(skf.split( df,df["stratify_group"])):

        print( f"\n{'=' * 60}")
        print( f"FOLD {fold + 1}/{NUM_FOLDS}")
        print(f"{'=' * 60}")
        train_df = df.iloc[ train_indices ].copy()

        val_df = df.iloc[val_indices ].copy()

        train_dataset =  ALTAMultiTaskDataset(dataframe=train_df,tokenizer=tokenizer,max_length=MAX_LENGTH)
        

        val_dataset = ALTAMultiTaskDataset(dataframe=val_df,tokenizer=tokenizer, max_length=MAX_LENGTH)
        

      
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

       
        model = DialectAwareMultiTaskDeBERTa(
            model_name=MODEL_NAME,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            dropout=0.1
        )
        # --------------------------------------------------
        # Initialize GENERAL LoRA from iSarcasmEval
        # --------------------------------------------------
        model.to(DEVICE)
        if USE_ISARCASM_PRETRAINING:
            state_before = {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
                if "lora_A" in name or "lora_B" in name
            }
            load_general_adapter_weights(
                model=model,
                checkpoint_path=GENERAL_ADAPTER_CHECKPOINT
            )

        
        model.rebuild_weighted_adapters()
        print(
            "\nAvailable adapters:"
        )

        print(
            model.encoder.peft_config.keys()
)
        # print_all_lora_adapter_norms(model)
        # changed = []

        # for name, param in model.named_parameters():

        #     if "lora_A" not in name and "lora_B" not in name:
        #         continue

        #     if name not in state_before:
        #         continue

        #     before = state_before[name]
        #     after = param.detach().cpu()

        #     difference = (
        #         after.float() - before.float()
        #     ).abs().max().item()

        #     if difference > 0:
        #         changed.append(
        #             (name, difference)
        #         )

        # print("\nChanged LoRA parameters:")
        # print(f"Count: {len(changed)}")

        # for name, diff in changed[:30]:
        #     print(
        #         f"{name}: max_difference={diff:.8f}"
        #     )
        # print_lora_adapter_norms(model)
        
        

        if NUM_GPUS > 1:
            print(
                f"Using DataParallel across "
                f"{NUM_GPUS} GPUs"
            )

            model = nn.DataParallel(model)
        else:
            print(
                f"Using single device: {DEVICE}"
            )

        

        # ==================================================
        # Sentiment loss
        # ==================================================

        sentiment_weights = compute_class_weights(
            train_df["sentiment"]
        ).to(DEVICE)

        sentiment_loss = nn.CrossEntropyLoss(
            weight=sentiment_weights
        )

        # ==================================================
        # Separate AU / UK sarcasm data
        # ==================================================

        au_train_df = train_df[
            train_df["variety"] == "en-AU"
        ]

        uk_train_df = train_df[
            train_df["variety"] == "en-UK"
        ]

        # ==================================================
        # Dialect-specific sarcasm weights
        # ==================================================

        au_sarcasm_weights = compute_class_weights(
            au_train_df["sarcasm"]
        ).to(DEVICE)

        uk_sarcasm_weights = compute_class_weights(
            uk_train_df["sarcasm"]
        ).to(DEVICE)

        print("\nSarcasm class weights:")

        print(
            f"en-AU: "
            f"negative={au_sarcasm_weights[0].item():.4f}, "
            f"positive={au_sarcasm_weights[1].item():.4f}"
        )

        print(
            f"en-UK: "
            f"negative={uk_sarcasm_weights[0].item():.4f}, "
            f"positive={uk_sarcasm_weights[1].item():.4f}"
        )

        # ==================================================
        # Separate focal losses
        # ==================================================

        au_sarcasm_loss = FocalLoss(
            alpha=au_sarcasm_weights,
            gamma=2.0
        )

        uk_sarcasm_loss = FocalLoss(
            alpha=uk_sarcasm_weights,
            gamma=2.0
        )




        
       
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        
        # steps_per_epoch = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS)
        steps_per_epoch = math.ceil(len(train_loader) / GRADIENT_ACCUMULATION_STEPS)

     
        steps_per_epoch = max(1, steps_per_epoch)
        
        total_training_steps = (steps_per_epoch * NUM_EPOCHS)

        warmup_steps = int( total_training_steps *  WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps= warmup_steps,
                num_training_steps=total_training_steps)
        


        best_score = -1.0

        best_epoch = 0

        epochs_without_improvement = 0


        for epoch in range(NUM_EPOCHS):

            print(
                f"\nEpoch {epoch + 1} / {NUM_EPOCHS}"
            )

            # ----------------------------------------------
            # Training
            # ----------------------------------------------

            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                sentiment_loss=sentiment_loss,
                au_sarcasm_loss=au_sarcasm_loss,
                uk_sarcasm_loss=uk_sarcasm_loss,
                sentiment_weight=0.4,
                sarcasm_weight=0.6
            )

            # ----------------------------------------------
            # Validation
            # ----------------------------------------------

            results = evaluate(
                model=model,
                dataloader=val_loader,
                sentiment_loss=sentiment_loss,
                au_sarcasm_loss=au_sarcasm_loss,
                uk_sarcasm_loss=uk_sarcasm_loss,
                sentiment_weight=0.4,
                sarcasm_weight=0.6
            )
            print("#########################################")
            print(f"Sentiment score:  {results['sentiment_score']:.4f}")
            print(f"Sentiment en-AU: {results['sentiment']['en-AU']:.4f}")
            print(f"Sentiment en-UK:{results['sentiment']['en-UK']:.4f}")
            print("--------")
            print(f"Sarcasm score:  {results['sarcasm_score']:.4f}"  )
            print(f"Sarcasm en-AU:   {results['sarcasm']['en-AU']:.4f}")
            print(f"Sarcasm en-UK:   {results['sarcasm']['en-UK']:.4f}")
            print("--------")
            
            print( f"Train Loss: {train_loss:.4f}")
            print( f"Validation Loss: {results['loss']:.4f}")

            sentiment_f1 = (
                results["sentiment"]["macro_f1"]
            )

            sarcasm_f1 = (
                results["sarcasm"]["macro_f1"]
            )

            print(f"Sentiment Macro F1: {sentiment_f1:.4f}" )

            print(f"Sarcasm Macro F1: {sarcasm_f1:.4f}")


            official_score = results["official_score"]
            print("official_score: "f"{official_score:.4f}")
            print("#########################################")
           

            # ----------------------------------------------
            # Check improvement
            # ----------------------------------------------

            if official_score > best_score + MIN_DELTA:

                best_score = official_score

                best_epoch = epoch + 1

                epochs_without_improvement = 0

                checkpoint_path = (
                    f"checkpoints/"
                    f"best_fold_"
                    f"{fold + 1}.pt"
                )

                # ------------------------------------------
                # Save underlying model if DataParallel
                # ------------------------------------------

                if isinstance(model, nn.DataParallel):

                    model_state_dict = (
                        model.module.state_dict()
                    )

                else:

                    model_state_dict = (
                        model.state_dict()
                    )

                torch.save(
                    {
                        "model_state_dict":
                            model_state_dict,

                        "best_score":
                            best_score,

                        "best_epoch":
                            best_epoch,

                        "fold":
                            fold + 1
                    },
                    checkpoint_path
                )

                print(
                    f"✓ New best model"
                )

                print(
                    f"✓ Saved best model "
                    f"→ {checkpoint_path}"
                )

            else:

                epochs_without_improvement += 1

                print(
                    f"No improvement for "
                    f"{epochs_without_improvement} "
                    f"epoch(s)."
                )

            # ----------------------------------------------
            # Early stopping
            # ----------------------------------------------

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nEarly stopping triggered."
                )

                print(
                    f"Best epoch: {best_epoch}"
                )

                print(
                    f"Best validation score: "
                    f"{best_score:.4f}"
                )

                break


        # ==================================================
        # FOLD RESULT
        # ==================================================

        fold_scores.append(
            best_score
        )

        print(
            f"\nBest Fold Score: "
            f"{best_score:.4f}"
        )

        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )


        # ==================================================
        # FREE FOLD MODEL
        # ==================================================

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

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

    # ==================================================
    # FINAL ENSEMBLE EVALUATION
    # ==================================================

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "FINAL ENSEMBLE EVALUATION"
    )

    print(
        "=" * 60
    )

    # ----------------------------------------------
    # Load separate evaluation dataset
    # ----------------------------------------------

    test_df = pd.read_csv(test_path)

    original_test_text = (
        test_df["text"].astype(str).copy()
    )

    test_df["text"] = (
        test_df["text"]
        .apply(preprocess_text)
    )

    test_changed_mask = (
        original_test_text != test_df["text"]
    )

    print(
        "\nValidation/Test preprocessing:"
    )

    print(
        f"Total samples:     {len(test_df)}"
    )

    print(
        f"Changed samples:   {test_changed_mask.sum()}"
    )

    print(
        f"Changed percentage: "
        f"{test_changed_mask.mean() * 100:.2f}%"
    )
    print(
        f"Evaluation samples: "
        f"{len(test_df)}"
    )

    print(
        "\nDialect distribution:"
    )

    print(
        test_df["variety"].value_counts()
    )


    test_dataset = ALTAMultiTaskDataset(
        dataframe=test_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # ==================================================
    # Load ALL fold checkpoints
    # ==================================================

    ensemble_models = []

    for fold in range(NUM_FOLDS):

        checkpoint_path = (
            f"checkpoints/"
            f"best_fold_{fold + 1}.pt"
        )

        if not os.path.exists(
            checkpoint_path
        ):

            raise FileNotFoundError(
                f"Required checkpoint not found: "
                f"{checkpoint_path}"
            )

        print(
            f"Loading Fold {fold + 1}: "
            f"{checkpoint_path}"
        )

        model = DialectAwareMultiTaskDeBERTa(
            model_name=MODEL_NAME,
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            dropout=0.1
        )

        model.to(DEVICE)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        
        if NUM_GPUS > 1:
            model = nn.DataParallel(model)

        model.eval()

        ensemble_models.append(
            model
        )

    print(
        f"\nLoaded "
        f"{len(ensemble_models)} "
        f"models for ensemble."
    )

    # ==================================================
    # Ensemble prediction
    # ==================================================

    ensemble_output = predict_ensemble(
        models=ensemble_models,
        dataloader=test_loader
    )

    # ==================================================
    # Calculate official ALTA score
    # ==================================================

    final_results = evaluate_final_test(

        varieties=
            ensemble_output["varieties"],

        sentiment_labels=
            ensemble_output["sentiment_labels"],

        sentiment_predictions=
            ensemble_output["sentiment_predictions"],

        sarcasm_labels=
            ensemble_output["sarcasm_labels"],

        sarcasm_predictions=
            ensemble_output["sarcasm_predictions"]
    )



    print(
        "\n"
        +
        "=" * 60
    )

    print(
        "FINAL ALTA ENSEMBLE RESULTS"
    )

    print(
        "=" * 60
    )

    print()

    print(
        f"f1-sentiment-en-AU: "
        f"{final_results['f1-sentiment-en-AU']:.4f}"
    )

    print(
        f"f1-sentiment-en-UK: "
        f"{final_results['f1-sentiment-en-UK']:.4f}"
    )

    print(
        f"f1-sarcasm-en-AU:   "
        f"{final_results['f1-sarcasm-en-AU']:.4f}"
    )

    print(
        f"f1-sarcasm-en-UK:   "
        f"{final_results['f1-sarcasm-en-UK']:.4f}"
    )

    print()

    print(
        f"FINAL SCORE: "
        f"{final_results['score']:.4f}"
    )

    # ==================================================
    # Create answer.csv
    # ==================================================

    answer_df = test_df[
        [
            "source",
            "variety",
            "text"
        ]
    ].copy()

    answer_df["sentiment"] = (
        ensemble_output[
            "sentiment_predictions"
        ]
    )

    answer_df["sarcasm"] = (
        ensemble_output[
            "sarcasm_predictions"
        ]
    )

    # ----------------------------------------------
    # Safety checks
    # ----------------------------------------------

    if len(answer_df) != len(test_df):

        raise RuntimeError(
            "Prediction count does not match "
            "test dataset size."
        )

    if not (
        answer_df["variety"].values
        ==
        test_df["variety"].values
    ).all():

        raise RuntimeError(
            "Dialect order was changed."
        )

    if not (
        answer_df["text"].values
        ==
        test_df["text"].values
    ).all():

        raise RuntimeError(
            "Text order was changed."
        )

    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    answer_path = "answer.csv"

    answer_df.to_csv(
        answer_path,
        index=False
    )

    print(
        "\n"
        +
        "=" * 60
    )

    print(
        f"Saved submission file:"
    )

    print(
        os.path.abspath(
            answer_path
        )
    )

 


    

if __name__ == "__main__":
    args = parse_args()
    main(args)