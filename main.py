import os
import random

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn

import json

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

train_path = f"train.csv" 
valid_path = f"valid.csv"
test_path = f"test.csv" 

# MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_NAME = "microsoft/deberta-v3-large"


# ============================================================
# iSarcasmEval GENERAL SARCASM PRETRAINING
# ============================================================

ISARCASM_TRAIN_PATH =  "iSarcasmEval_train.csv"
ISARCASM_TEST_BINARY_PATH = "iSarcasmEvalTest_task_A_En_test.csv"
ISARCASM_TEST_ADDITIONAL_PATH = "iSarcasmEvalTest_task_B_En_test.csv"

USE_ISARCASM_PRETRAINING = False
USE_REDDIT_PRETRAINING =  True 

USE_REDDIT_DATASET = False 


# Reddit shared-task sarcasm data
REDDIT_SARCASM_PATH = "sarcasm_detection_shared_task_reddit_training.jsonl"
REDDIT_SARCASM_PATH_2 = "sarcasm_detection_shared_task_reddit_testing.jsonl"

# Start with 2,000 Reddit samples for a controlled experiment.
# The dataset is approximately balanced, so we take 1,000 per class.
REDDIT_PRETRAIN_SAMPLES = 4400

GENERAL_ADAPTER_CHECKPOINT = (
    "checkpoints/"
)

ALTA_CHECKPOINT_DIR = (
    "checkpoints/task_specific"
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
def check_model_finite(model):
    print("\n" + "=" * 70)
    print("FINITE VALUE CHECK")
    print("=" * 70)

    bad = []

    for name, param in model.named_parameters():

        if not torch.isfinite(param).all():

            bad.append(name)

            print(
                f"NON-FINITE: {name}"
            )

    if len(bad) == 0:
        print("✓ All model parameters are finite.")
    else:
        print(
            f"✗ Found {len(bad)} parameters "
            f"containing NaN/Inf."
        )

    print("=" * 70)

def initialize_sarcasm_adapters_from_pretrained(
    model,
    checkpoint_path,
):
    """
    Initialize both ALTA sarcasm adapters from the pretrained
    iSarcasm general LoRA adapter.

        iSarcasm general
              │
        ┌─────┴─────┐
        ▼           ▼
    sarcasm_au   sarcasm_uk

    Sentiment adapters are NOT modified.
    """

    import os
    import torch

    device = next(model.parameters()).device

    # ============================================================
    # Resolve checkpoint
    # ============================================================

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint path does not exist:\n"
            f"{checkpoint_path}"
        )

    if os.path.isdir(checkpoint_path):

        checkpoint_file = os.path.join(
            checkpoint_path,
            "best_general_adapter.pt",
        )

        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(
                "Could not find pretrained checkpoint:\n"
                f"{checkpoint_file}\n\n"
                f"Directory contents:\n"
                f"{os.listdir(checkpoint_path)}"
            )

    else:
        checkpoint_file = checkpoint_path

    print(
        "\nLoading pretrained sarcasm adapter from:"
    )
    print(checkpoint_file)

    # ============================================================
    # Load checkpoint
    # ============================================================

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

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
            pretrained_state = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format: "
            f"{type(checkpoint)}"
        )

    # ============================================================
    # Current model state
    # ============================================================

    current_state = model.state_dict()

    transferred_au = 0
    transferred_uk = 0
    skipped = 0

    au_examples = []
    uk_examples = []

    # ============================================================
    # Transfer
    # ============================================================

    for source_key, source_value in pretrained_state.items():

        # We only want the pretrained general LoRA tensors.
        if "lora_A.general." not in source_key and \
           "lora_B.general." not in source_key:
            continue

        # --------------------------------------------------------
        # Create exact target names by replacing ONLY the adapter
        # name.
        # --------------------------------------------------------

        au_key = (
            source_key
            .replace(
                ".lora_A.general.",
                ".lora_A.sarcasm_au.",
            )
            .replace(
                ".lora_B.general.",
                ".lora_B.sarcasm_au.",
            )
        )

        uk_key = (
            source_key
            .replace(
                ".lora_A.general.",
                ".lora_A.sarcasm_uk.",
            )
            .replace(
                ".lora_B.general.",
                ".lora_B.sarcasm_uk.",
            )
        )

        # --------------------------------------------------------
        # Sanity check
        # --------------------------------------------------------

        source_is_lora = (
            "lora_A.general" in source_key
            or "lora_B.general" in source_key
        )

        if not source_is_lora:
            continue

        # ========================================================
        # AU
        # ========================================================

        if au_key in current_state:

            if (
                current_state[au_key].shape
                == source_value.shape
            ):

                current_state[au_key] = (
                    source_value.to(
                        device=current_state[
                            au_key
                        ].device,
                        dtype=current_state[
                            au_key
                        ].dtype,
                    )
                )

                transferred_au += 1

                if len(au_examples) < 5:
                    au_examples.append(
                        (
                            source_key,
                            au_key,
                        )
                    )

            else:
                skipped += 1

        else:
            skipped += 1

        # ========================================================
        # UK
        # ========================================================

        if uk_key in current_state:

            if (
                current_state[uk_key].shape
                == source_value.shape
            ):

                current_state[uk_key] = (
                    source_value.to(
                        device=current_state[
                            uk_key
                        ].device,
                        dtype=current_state[
                            uk_key
                        ].dtype,
                    )
                )

                transferred_uk += 1

                if len(uk_examples) < 5:
                    uk_examples.append(
                        (
                            source_key,
                            uk_key,
                        )
                    )

            else:
                skipped += 1

        else:
            skipped += 1

    # ============================================================
    # Load modified state
    # ============================================================

    model.load_state_dict(
        current_state,
        strict=False,
    )

    # ============================================================
    # Report
    # ============================================================

    print()
    print("=" * 70)
    print(
        "iSARCASM → ALTA SARCASM ADAPTER INITIALIZATION"
    )
    print("=" * 70)

    print(
        f"Source LoRA tensors : "
        f"{len([k for k in pretrained_state if 'lora_A.general.' in k or 'lora_B.general.' in k])}"
    )

    print(
        f"sarcasm_au tensors  : "
        f"{transferred_au}"
    )

    print(
        f"sarcasm_uk tensors  : "
        f"{transferred_uk}"
    )

    print(
        f"Skipped             : "
        f"{skipped}"
    )

    print("=" * 70)

    # ------------------------------------------------------------
    # Show a few mappings so we can verify the names.
    # ------------------------------------------------------------

    print("\nExample AU mappings:")

    for source_key, target_key in au_examples:
        print(
            f"  {source_key}\n"
            f"    -> {target_key}"
        )

    print("\nExample UK mappings:")

    for source_key, target_key in uk_examples:
        print(
            f"  {source_key}\n"
            f"    -> {target_key}"
        )

    print()

    # ============================================================
    # Required sanity check
    # ============================================================

    # ============================================================
    # Determine expected transfer count dynamically
    # ============================================================
    # The checkpoint stores two tensors per target module
    # (lora_A and lora_B). Both are copied.
    expected_per_adapter = len(
        [
            k for k in pretrained_state
            if "lora_A.general." in k or "lora_B.general." in k
        ]
    )

    if expected_per_adapter == 0:
        raise RuntimeError(
            "No general LoRA tensors were found in the "
            "pretrained checkpoint. Did you load the right file?"
        )

    # ============================================================
    # Sanity checks
    # ============================================================

    if transferred_au != expected_per_adapter:
        raise RuntimeError(
            f"Expected {expected_per_adapter} LoRA tensors "
            f"for sarcasm_au, but transferred {transferred_au}."
        )

    if transferred_uk != expected_per_adapter:
        raise RuntimeError(
            f"Expected {expected_per_adapter} LoRA tensors "
            f"for sarcasm_uk, but transferred {transferred_uk}."
        )

    print(
        f"✓ {expected_per_adapter}/{expected_per_adapter} "
        f"iSarcasm LoRA tensors copied to sarcasm_au."
    )

    print(
        f"✓ {expected_per_adapter}/{expected_per_adapter} "
        f"iSarcasm LoRA tensors copied to sarcasm_uk."
    )

    return model
def force_lora_parameters_to_float32(model):
    """
    Force all ALTA LoRA parameters to float32.

    This prevents unstable optimizer updates when LoRA
    parameters are stored in float16/bfloat16.
    """

    converted = 0

    for name, param in model.named_parameters():

        if (
            "lora_A" not in name
            and "lora_B" not in name
        ):
            continue

        if param.dtype != torch.float32:

            param.data = param.data.float()

            converted += 1

    print(
        f"✓ Converted {converted} LoRA parameter tensors "
        f"to float32."
    )    
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
            # print_adapter_gradient_stats(model)
            # print(
            #     f"Before optimizer step: "
            #     f"loss={total_batch_loss.item():.6f}"
            # )
            # ============================================================
            # DEBUG: CHECK PARAMETERS AFTER OPTIMIZER STEP
            # ============================================================

            bad_parameters = []

            for name, param in model.named_parameters():

                if not torch.isfinite(param).all():

                    bad_parameters.append(name)

            # print(
            #     f"After optimizer step: "
            #     f"bad_parameters={len(bad_parameters)}"
            # )

            if bad_parameters:

                # print("First non-finite parameters:")

                # for name in bad_parameters[:20]:
                #     print(
                #         f"  {name}"
                #     )

                raise RuntimeError(
                    "Non-finite parameter detected after optimizer.step()."
                )

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
def load_reddit_sarcasm_data(
    jsonl_path,
    max_samples=None,
    seed=42,
):
    """
    Load the Reddit shared-task sarcasm dataset.

    Output schema:
        tweet
        sarcastic
        rephrase
        sarcasm

    Reddit provides only binary sarcasm supervision:
        rephrase = ""
        sarcasm = NaN

    max_samples=None means use ALL valid Reddit examples.
    """

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Reddit sarcasm dataset not found:\n{jsonl_path}"
        )

    rows = []

    with open(jsonl_path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

            except json.JSONDecodeError as e:
                # print(
                #     f"Warning: could not parse line "
                #     f"{line_number}: {e}"
                # )
                continue

            label = str(
                item.get("label", "")
            ).strip().upper()

            response = str(
                item.get("response", "")
            ).strip()

            context = item.get(
                "context",
                [],
            )

            if not isinstance(context, list):
                context = [str(context)]

            context = [
                str(x).strip()
                for x in context
                if str(x).strip()
            ]

            if label not in {
                "SARCASM",
                "NOT_SARCASM",
            }:
                continue

            if not response:
                continue

            # Keep target response first because
            # MAX_LENGTH=128 may truncate the end.
            recent_context = context[-2:]

            if recent_context:

                context_text = "\n".join(
                    f"Context {i + 1}: {text}"
                    for i, text in enumerate(
                        recent_context
                    )
                )

                combined_text = (
                    f"Response: {response}\n"
                    f"{context_text}"
                )

            else:

                combined_text = (
                    f"Response: {response}"
                )

            sarcastic = (
                1
                if label == "SARCASM"
                else 0
            )

            rows.append(
                {
                    "tweet": combined_text,
                    "sarcastic": sarcastic,
                    "rephrase": "",
                    "sarcasm": np.nan,
                }
            )

    reddit_df = pd.DataFrame(rows)

    if reddit_df.empty:
        raise ValueError(
            "No valid Reddit sarcasm examples were loaded."
        )

    # --------------------------------------------------
    # Remove exact duplicate texts
    # --------------------------------------------------

    before_dedup = len(reddit_df)

    reddit_df["tweet_normalized"] = (
        reddit_df["tweet"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    reddit_df = (
        reddit_df
        .drop_duplicates(
            subset=["tweet_normalized"],
            keep="first",
        )
        .drop(
            columns=["tweet_normalized"]
        )
        .reset_index(drop=True)
    )

    duplicate_count = (
        before_dedup - len(reddit_df)
    )

    # --------------------------------------------------
    # Optional maximum sample limit
    # --------------------------------------------------

    if (
        max_samples is not None
        and max_samples > 0
        and len(reddit_df) > max_samples
    ):

        reddit_df = reddit_df.sample(
            n=max_samples,
            random_state=seed,
        ).reset_index(drop=True)

    # --------------------------------------------------
    # Shuffle
    # --------------------------------------------------

    reddit_df = reddit_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("REDDIT SARCASM PRETRAINING DATA")
    print("=" * 60)

    print(
        f"Original valid Reddit samples: "
        f"{before_dedup}"
    )

    print(
        f"Duplicate Reddit examples removed: "
        f"{duplicate_count}"
    )

    print(
        f"Final Reddit samples: "
        f"{len(reddit_df)}"
    )

    print("\nReddit binary distribution:")

    print(
        reddit_df["sarcastic"]
        .value_counts()
        .sort_index()
    )

    print("\nExample:")

    print(
        reddit_df.iloc[0]["tweet"][:1000]
    )

    print(
        f"Label: "
        f"{reddit_df.iloc[0]['sarcastic']}"
    )

    return reddit_df

def pretrain_reddit_general_adapter(
    reddit_jsonl_path=REDDIT_SARCASM_PATH,
    reddit_jsonl_path_2=REDDIT_SARCASM_PATH_2,
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
    seed=SEED,
    num_workers=2,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ============================================================
    # REPRODUCIBILITY
    # ============================================================

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ============================================================
    # LOAD ALL REDDIT DATA
    # ============================================================

    reddit_df_train = load_reddit_sarcasm_data(
        jsonl_path=reddit_jsonl_path,
        max_samples=None,
        seed=seed,
    )
    reddit_df_test = load_reddit_sarcasm_data(
            jsonl_path=reddit_jsonl_path_2,
            max_samples=None,
            seed=seed,
        )

    reddit_df = pd.concat([reddit_df_train, reddit_df_test], ignore_index=True)
    reddit_df = reddit_df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)

    print("\n" + "=" * 60)
    print("REDDIT-ONLY GENERAL ADAPTER PRETRAINING")
    print("=" * 60)

    print(
        f"Total Reddit samples: "
        f"{len(reddit_df)}"
    )

    # ============================================================
    # SPLIT REDDIT INTO TRAIN / VALIDATION
    # ============================================================

    train_df, val_df = train_test_split(
        reddit_df,
        test_size=val_size,
        random_state=seed,
        stratify=reddit_df["sarcastic"],
    )

    train_df = (
        train_df
        .reset_index(drop=True)
    )

    val_df = (
        val_df
        .reset_index(drop=True)
    )

    print(
        f"Reddit training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Reddit validation samples: "
        f"{len(val_df)}"
    )

    print("\nTraining distribution:")

    print(
        train_df["sarcastic"]
        .value_counts()
        .sort_index()
    )

    print("\nValidation distribution:")

    print(
        val_df["sarcastic"]
        .value_counts()
        .sort_index()
    )

    # ============================================================
    # TOKENIZER
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # ============================================================
    # DATASETS
    # ============================================================

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

    # ============================================================
    # DATALOADERS
    # ============================================================

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

    # ============================================================
    # DEVICE
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ============================================================
    # MODEL
    # ============================================================

    model = GeneralSarcasmPretrainModel(
        model_name=model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        dropout=0.1,
    )

    model.to(device)

    model.encoder.print_trainable_parameters()

    # ============================================================
    # BINARY CLASS WEIGHTS
    # ============================================================

    binary_counts = (
        train_df["sarcastic"]
        .value_counts()
        .sort_index()
    )

    n_negative = int(
        binary_counts.get(0, 1)
    )

    n_positive = int(
        binary_counts.get(1, 1)
    )

    class_weights = torch.tensor(
        [
            1.0,
            n_negative / max(
                n_positive,
                1,
            ),
        ],
        dtype=torch.float32,
        device=device,
    )

    binary_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # print(
    #     f"\nBinary class weights: "
    #     f"{class_weights.detach().cpu().numpy()}"
    # )

    # ============================================================
    # NO FINE-GRAINED / REPHRASE SUPERVISION
    # ============================================================

    print(
        "\nReddit provides only binary sarcasm labels."
    )

    print(
        "Rephrase loss: DISABLED"
    )

    print(
        "Fine sarcasm loss: DISABLED"
    )

    # ============================================================
    # OPTIMIZER
    # ============================================================

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # ============================================================
    # SCHEDULER
    # ============================================================

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
        steps_per_epoch
        * num_epochs
    )

    warmup_steps = int(
        total_steps
        * warmup_ratio
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ============================================================
    # TRAINING
    # ============================================================

    best_f1 = -1.0
    best_epoch = -1

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Reddit has ONLY binary labels.
        #
        # Therefore:
        #   binary_weight = 1.0
        #   rephrase_weight = 0.0
        #   fine_weight = 0.0
        #
        # --------------------------------------------------------

        train_metrics = pretrain_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            binary_loss_fn=binary_loss_fn,
            fine_pos_weight=1.0,
            contrastive_loss_fn=None,
            binary_weight=1.0,
            rephrase_weight=0.0,
            fine_weight=0.0,
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
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

        # ========================================================
        # SAVE BEST REDDIT ADAPTER
        # ========================================================

        if (
            val_metrics["macro_f1"]
            > best_f1
        ):

            best_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            checkpoint_path = os.path.join(
                output_dir,
                "best_general_adapter.pt",
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_f1":
                        best_f1,

                    "best_epoch":
                        best_epoch,

                    "model_name":
                        model_name,

                    "pretraining_dataset":
                        "reddit_only",

                    "training_samples":
                        len(train_df),

                    "validation_samples":
                        len(val_df),
                },
                checkpoint_path,
            )

            print(
                f"✓ Saved best Reddit model → "
                f"{checkpoint_path}"
            )

    # ============================================================
    # FINAL SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("REDDIT-ONLY PRETRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best validation Macro F1: "
        f"{best_f1:.4f}"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_df)}"
    )

    print(
        f"Checkpoint: "
        f"{os.path.join(output_dir, 'best_general_adapter.pt')}"
    )

    return model, tokenizer

def load_additional_isarcasm_test_data(
    binary_test_csv=None,
    fine_test_csv=None,
):
    """
    Load additional labeled iSarcasm data.

    binary_test_csv:
        columns: test, sarcastic

    fine_test_csv:
        columns:
            text, sarcasm, irony, satire,
            understatement, overstatement,
            rhetorical_question

    Returns a canonical dataframe with:
        tweet
        sarcastic
        sarcasm
        rephrase

    The five auxiliary category columns are retained in the dataframe
    for future experiments, but are NOT currently used as losses.
    """

    binary_df = None
    fine_df = None

    # ============================================================
    # Binary test file
    # ============================================================

    if binary_test_csv is not None:

        binary_df = pd.read_csv(binary_test_csv)

        required = ["text", "sarcastic"]

        missing = [
            c for c in required
            if c not in binary_df.columns
        ]

        if missing:
            raise ValueError(
                f"Binary iSarcasm test file is missing: {missing}"
            )

        binary_df = binary_df.rename(
            columns={
                "text": "tweet",
            }
        )

        binary_df["tweet"] = (
            binary_df["tweet"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        binary_df["sarcastic"] = pd.to_numeric(
            binary_df["sarcastic"],
            errors="coerce",
        )

        binary_df = binary_df[
            binary_df["tweet"].ne("")
            & binary_df["sarcastic"].notna()
        ].copy()

        binary_df["sarcastic"] = (
            binary_df["sarcastic"]
            .astype(int)
        )

    # ============================================================
    # Fine-grained test file
    # ============================================================

    if fine_test_csv is not None:

        fine_df = pd.read_csv(fine_test_csv)

        required = [
            "text",
            "sarcasm",
            "irony",
            "satire",
            "understatement",
            "overstatement",
            "rhetorical_question",
        ]

        missing = [
            c for c in required
            if c not in fine_df.columns
        ]

        if missing:
            raise ValueError(
                f"Fine-grained iSarcasm test file is missing: {missing}"
            )

        fine_df = fine_df.rename(
            columns={
                "text": "tweet",
            }
        )

        fine_df["tweet"] = (
            fine_df["tweet"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        fine_df = fine_df[
            fine_df["tweet"].ne("")
        ].copy()

        # Convert the relevant fine-grained labels.
        for col in [
            "sarcasm",
            "irony",
            "satire",
            "understatement",
            "overstatement",
            "rhetorical_question",
        ]:
            fine_df[col] = pd.to_numeric(
                fine_df[col],
                errors="coerce",
            )

    # ============================================================
    # Nothing supplied
    # ============================================================

    if binary_df is None and fine_df is None:
        return pd.DataFrame()

    # ============================================================
    # Merge the two test files
    # ============================================================

    if binary_df is not None and fine_df is not None:

        merged = pd.merge(
            binary_df,
            fine_df,
            on="tweet",
            how="outer",
            suffixes=("", "_fine"),
        )

        # Prefer the binary-test sarcastic label.
        if "sarcastic_fine" in merged.columns:

            merged["sarcastic"] = (
                merged["sarcastic"]
                .fillna(
                    merged["sarcastic_fine"]
                )
            )

            merged.drop(
                columns=["sarcastic_fine"],
                inplace=True,
            )

    elif binary_df is not None:

        merged = binary_df.copy()

    else:

        merged = fine_df.copy()

    # ============================================================
    # Canonical columns required by current dataset
    # ============================================================

    if "sarcastic" not in merged.columns:

        # If only the fine-grained file is available,
        # infer binary sarcasm from its "sarcasm" annotation.
        if "sarcasm" in merged.columns:

            merged["sarcastic"] = (
                merged["sarcasm"]
                .fillna(0)
                .astype(int)
            )

        else:
            raise ValueError(
                "Could not construct binary sarcastic labels."
            )

    # Rephrase does not exist in these files.
    merged["rephrase"] = ""

    # Ensure fine sarcasm exists.
    if "sarcasm" not in merged.columns:

        merged["sarcasm"] = np.nan

    # ============================================================
    # Remove exact duplicate tweets
    # ============================================================

    merged = (
        merged
        .drop_duplicates(
            subset=["tweet"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        "\nAdditional iSarcasm labeled data:"
    )

    print(
        f"Samples: {len(merged)}"
    )

    print(
        "Binary sarcasm distribution:\n",
        merged["sarcastic"].value_counts(
            dropna=False
        )
    )

    print(
        "Fine sarcasm availability:",
        merged["sarcasm"].notna().sum()
    )

    return merged

def remove_duplicate_tweets_against_reference(
    source_df,
    reference_df,
):
    """
    Remove examples from source_df whose normalized text already
    exists in reference_df.
    """

    source = source_df.copy()
    reference = reference_df.copy()

    source["_normalized_text"] = (
        source["tweet"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    reference_texts = set(
        reference["tweet"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    before = len(source)

    source = source[
        ~source["_normalized_text"].isin(reference_texts)
    ].copy()

    source = source.drop(
        columns=["_normalized_text"]
    )

    removed = before - len(source)

    return source, removed


def sarcasm_pretrain_general_adapter(
    train_csv=ISARCASM_TRAIN_PATH,
    additional_binary_test_csv=None,
    additional_fine_test_csv=None,
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

   
    # Load original iSarcasm training data

    df = pd.read_csv(train_csv)

    required_columns = [
        "tweet",
        "sarcastic",
        "rephrase",
        "sarcasm",
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        f"Original iSarcasm training samples: {len(df)}"
    )

    # --------------------------------------------------
    # Split ORIGINAL train data first
    #
    # Important:
    # validation remains completely untouched by the
    # additional labeled test data.
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
        f"Original train split: {len(train_df)}"
    )

    print(
        f"Original validation split: {len(val_df)}"
    )

    # --------------------------------------------------
    # Load additional iSarcasm data
    # --------------------------------------------------

    additional_df = load_additional_isarcasm_test_data(
        binary_test_csv=ISARCASM_TEST_BINARY_PATH,
        fine_test_csv=ISARCASM_TEST_ADDITIONAL_PATH,
    )

    # --------------------------------------------------
    # Load Reddit shared-task sarcasm data
    # --------------------------------------------------

    # reddit_df = load_reddit_sarcasm_data(
    #     jsonl_path=REDDIT_SARCASM_PATH,
    #     max_samples=REDDIT_PRETRAIN_SAMPLES,
    #     seed=seed,
    # )
    

    if USE_REDDIT_DATASET:
        reddit_df = load_reddit_sarcasm_data(
            jsonl_path=REDDIT_SARCASM_PATH,
            max_samples=REDDIT_PRETRAIN_SAMPLES,
            seed=seed,
        )
    else:
        reddit_df = pd.DataFrame(
        columns=[
            "tweet",
            "sarcasm",
            "sarcastic",
            "rephrase",
        ]
    )
        
    print(f"Reddit dataset: {'USED' if USE_REDDIT_DATASET else 'SKIPPED'}")
    print(f"Reddit samples: {len(reddit_df)}")

    # ============================================================
    # ADDITIONAL iSARCASM
    # ============================================================

    if len(additional_df) > 0:

        # Make sure the canonical columns exist.
        for col in [
            "tweet",
            "sarcastic",
            "rephrase",
            "sarcasm",
        ]:
            if col not in additional_df.columns:

                if col == "rephrase":
                    additional_df[col] = ""

                elif col == "sarcasm":
                    additional_df[col] = np.nan

                else:
                    raise ValueError(
                        f"Additional data is missing '{col}'"
                    )

        # Keep only the columns used by ISarcasmDataset.
        additional_df = additional_df[
            [
                "tweet",
                "sarcastic",
                "rephrase",
                "sarcasm",
            ]
        ].copy()

        # Normalize text.
        additional_df["tweet"] = (
            additional_df["tweet"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Remove duplicates against the ORIGINAL iSarcasm
        # training split.
        additional_df, removed_additional = (
            remove_duplicate_tweets_against_reference(
                additional_df,
                train_df,
            )
        )

        print(
            f"\nAdditional iSarcasm data added: "
            f"{len(additional_df)}"
        )

        print(
            f"Additional iSarcasm duplicates removed: "
            f"{removed_additional}"
        )

        # Add additional iSarcasm to training.
        train_df = pd.concat(
            [
                train_df,
                additional_df,
            ],
            ignore_index=True,
        )

    else:

        print(
            "\nNo additional iSarcasm test data supplied."
        )


    # ============================================================
    # REDDIT
    # ============================================================

    # Remove Reddit examples that duplicate ANY text already
    # present in the iSarcasm training data.
    reddit_df, removed_reddit = (
        remove_duplicate_tweets_against_reference(
            reddit_df,
            train_df,
        )
    )

    print(
        f"\nReddit data added: {len(reddit_df)}"
    )

    print(
        f"Reddit duplicates removed: {removed_reddit}"
    )

    # ------------------------------------------------------------
    # Add Reddit to the SAME training dataframe.
    #
    # Reddit contributes:
    #   sarcastic -> binary sarcasm loss
    #
    # Reddit has:
    #   rephrase = ""
    #   sarcasm   = NaN
    #
    # Therefore the existing masking in pretrain_one_epoch()
    # automatically excludes Reddit from those two objectives.
    # ------------------------------------------------------------

    reddit_df = reddit_df[
        [
            "tweet",
            "sarcastic",
            "rephrase",
            "sarcasm",
        ]
    ].copy()

    train_df = pd.concat(
        [
            train_df,
            reddit_df,
        ],
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Final duplicate removal
    # ------------------------------------------------------------

    before_final_dedup = len(train_df)

    train_df["tweet"] = (
        train_df["tweet"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    train_df = (
        train_df
        .drop_duplicates(
            subset=["tweet"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    final_duplicates_removed = (
        before_final_dedup - len(train_df)
    )

    # ------------------------------------------------------------
    # Shuffle final training data
    # ------------------------------------------------------------

    train_df = train_df.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

    # ============================================================
    # FINAL DATASET SUMMARY
    # ============================================================

    print("\n" + "=" * 60)
    print("FINAL GENERAL PRETRAINING DATA")
    print("=" * 60)

    print(
        f"Original iSarcasm train: "
        f"{len(train_df) - len(reddit_df) - len(additional_df)}"
    )

    print(
        f"Additional iSarcasm:    "
        f"{len(additional_df)}"
    )

    print(
        f"Reddit:                 "
        f"{len(reddit_df)}"
    )

    print(
        f"Final duplicates removed: "
        f"{final_duplicates_removed}"
    )

    print(
        f"Final training size: "
        f"{len(train_df)}"
    )

    print(
        f"Validation size: "
        f"{len(val_df)}"
    )

    print("\nFinal binary sarcasm distribution:")
    print(
        train_df["sarcastic"]
        .value_counts()
        .sort_index()
    )

  
    print(
        f"\nFinal training size: {len(train_df)}"
    )

    print(
        f"Validation size: {len(val_df)}"
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

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------
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
        "cuda"
        if torch.cuda.is_available()
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

    n_negative = int(
        binary_counts.get(0, 1)
    )

    n_positive = int(
        binary_counts.get(1, 1)
    )

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

    # print(
    #     f"Fine sarcasm pos_weight: "
    #     f"{fine_pos_weight:.4f}"
    # )

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
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        eps=1e-6,
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------
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

    for epoch in range(
        1,
        num_epochs + 1
    ):

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
            gradient_accumulation_steps=(
                gradient_accumulation_steps
            ),
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
        if (
            val_metrics["macro_f1"]
            > best_f1
        ):

            best_f1 = (
                val_metrics["macro_f1"]
            )

            best_epoch = epoch

            checkpoint_path = os.path.join(
                output_dir,
                "best_general_adapter.pt",
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "best_f1":
                        best_f1,

                    "best_epoch":
                        best_epoch,

                    "model_name":
                        model_name,
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
    all_sarcasm_probs = []
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
        sarcasm_positive_probs = torch.softmax(
            sarcasm_logits,
            dim=1
        )[:, 1]
        all_sarcasm_probs.extend(sarcasm_positive_probs.cpu().numpy())

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
    print(
        f"Sarcasm positive probability mean: "
        f"{np.mean(all_sarcasm_probs):.4f}"
    )

    print(
        f"Sarcasm positive probability std: "
        f"{np.std(all_sarcasm_probs):.4f}"
    )
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
# @torch.no_grad()
# def predict_ensemble(
#     models,
#     dataloader
# ):
    
#     for model in models:
#         model.eval()

#     varieties = []

#     sentiment_labels_all = []
#     sarcasm_labels_all = []

#     sentiment_logits_all_models = []
#     sarcasm_logits_all_models = []

    
#     for model_index, model in enumerate(models):

#         sentiment_logits_model = []
#         sarcasm_logits_model = []

#         sentiment_labels_model = []
#         sarcasm_labels_model = []

#         varieties_model = []

#         for batch in dataloader:

#             input_ids = batch[
#                 "input_ids"
#             ].to(DEVICE)

#             attention_mask = batch[
#                 "attention_mask"
#             ].to(DEVICE)

#             variety_ids = batch[
#                 "variety_ids"
#             ].to(DEVICE)

#             sentiment_labels = batch[
#                 "sentiment_labels"
#             ].to(DEVICE)

#             sarcasm_labels = batch[
#                 "sarcasm_labels"
#             ].to(DEVICE)

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 variety_ids=variety_ids
#             )

#             sentiment_logits_model.append(
#                 outputs[
#                     "sentiment_logits"
#                 ].cpu()
#             )

#             sarcasm_logits_model.append(
#                 outputs[
#                     "sarcasm_logits"
#                 ].cpu()
#             )

#             sentiment_labels_model.extend(
#                 sentiment_labels.cpu().numpy()
#             )

#             sarcasm_labels_model.extend(
#                 sarcasm_labels.cpu().numpy()
#             )

#             varieties_model.extend(
#                 batch["variety"]
#             )

#         sentiment_logits_model = torch.cat(
#             sentiment_logits_model,
#             dim=0
#         )

#         sarcasm_logits_model = torch.cat(
#             sarcasm_logits_model,
#             dim=0
#         )

#         sentiment_logits_all_models.append(
#             sentiment_logits_model
#         )

#         sarcasm_logits_all_models.append(
#             sarcasm_logits_model
#         )

#         if model_index == 0:

#             sentiment_labels_all = (
#                 sentiment_labels_model
#             )

#             sarcasm_labels_all = (
#                 sarcasm_labels_model
#             )

#             varieties = varieties_model

 

#     sentiment_logits_ensemble = torch.stack(
#         sentiment_logits_all_models,
#         dim=0
#     ).mean(dim=0)

#     sarcasm_logits_ensemble = torch.stack(
#         sarcasm_logits_all_models,
#         dim=0
#     ).mean(dim=0)


#     sentiment_predictions = torch.argmax(
#         sentiment_logits_ensemble,
#         dim=1
#     ).numpy()

#     sarcasm_predictions = torch.argmax(
#         sarcasm_logits_ensemble,
#         dim=1
#     ).numpy()

#     return {
#         "varieties": varieties,

#         "sentiment_labels":
#             sentiment_labels_all,

#         "sentiment_predictions":
#             sentiment_predictions,

#         "sarcasm_labels":
#             sarcasm_labels_all,

#         "sarcasm_predictions":
#             sarcasm_predictions
#     }


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
                outputs["sentiment_logits"].cpu()
            )

            sarcasm_logits_model.append(
                outputs["sarcasm_logits"].cpu()
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

    # ==================================================
    # Average logits across folds
    # ==================================================

    sentiment_logits_ensemble = torch.stack(
        sentiment_logits_all_models,
        dim=0
    ).mean(dim=0)

    sarcasm_logits_ensemble = torch.stack(
        sarcasm_logits_all_models,
        dim=0
    ).mean(dim=0)

    # ==================================================
    # Probabilities
    # ==================================================

    sentiment_probabilities = torch.softmax(
        sentiment_logits_ensemble,
        dim=1
    )

    sarcasm_probabilities = torch.softmax(
        sarcasm_logits_ensemble,
        dim=1
    )

    # Probability of positive class
    sentiment_positive_probs = (
        sentiment_probabilities[:, 1]
        .numpy()
    )

    sarcasm_positive_probs = (
        sarcasm_probabilities[:, 1]
        .numpy()
    )

    # ==================================================
    # Default predictions
    # ==================================================

    sentiment_predictions = torch.argmax(
        sentiment_logits_ensemble,
        dim=1
    ).numpy()

    sarcasm_predictions = torch.argmax(
        sarcasm_logits_ensemble,
        dim=1
    ).numpy()

    return {

        "varieties":
            varieties,

        "sentiment_labels":
            np.asarray(sentiment_labels_all),

        "sentiment_predictions":
            sentiment_predictions,

        "sarcasm_labels":
            np.asarray(sarcasm_labels_all),

        "sarcasm_predictions":
            sarcasm_predictions,

        # NEW
        "sentiment_positive_probs":
            sentiment_positive_probs,

        "sarcasm_positive_probs":
            sarcasm_positive_probs,

        # Keep logits too
        "sentiment_logits":
            sentiment_logits_ensemble.numpy(),

        "sarcasm_logits":
            sarcasm_logits_ensemble.numpy()
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

        # print(
        #     f"{name:<100} "
        #     f"norm={param.detach().float().norm().item():.8f}"
        # )

    # print("=" * 70)
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


def enable_all_lora_gradients(model):

    adapters = [
        "sentiment_au",
        "sentiment_uk",
        "sarcasm_au",
        "sarcasm_uk",
    ]

    counts = {
        adapter: 0
        for adapter in adapters
    }

    for name, param in model.named_parameters():

        matched = False

        for adapter in adapters:

            if (
                f"lora_A.{adapter}." in name
                or f"lora_B.{adapter}." in name
            ):
                param.requires_grad_(True)
                counts[adapter] += param.numel()
                matched = True
                break

    # print("\n" + "=" * 70)
    # print("LORA TRAINABILITY CHECK")
    # print("=" * 70)

    # for adapter in adapters:
    #     print(
    #         f"{adapter:15s}: "
    #         f"{counts[adapter]:,} parameters"
    #     )

    # print("=" * 70)

def print_adapter_gradient_stats(model):

    adapters = [
        "sentiment_au",
        "sentiment_uk",
        "sarcasm_au",
        "sarcasm_uk",
    ]

    # print("\n" + "=" * 70)
    # print("ADAPTER GRADIENT STATISTICS")
    # print("=" * 70)

    for adapter in adapters:

        total_norm_sq = 0.0
        grad_tensors = 0
        missing_gradients = 0
        nonzero_gradients = 0

        for name, param in model.named_parameters():

            if (
                f"lora_A.{adapter}." not in name
                and f"lora_B.{adapter}." not in name
            ):
                continue

            if param.grad is None:
                missing_gradients += 1
                continue

            grad_tensors += 1

            grad_norm = (
                param.grad.detach()
                .float()
                .norm()
                .item()
            )

            total_norm_sq += grad_norm ** 2

            if grad_norm > 0:
                nonzero_gradients += 1

        total_norm = (
            total_norm_sq ** 0.5
        )

        # print( f"{adapter:15s} | norm={total_norm:.6e} | grad={grad_tensors} | missing={missing_gradients} | nonzero={nonzero_gradients}")

    # print("=" * 70)


@torch.no_grad()
def get_sarcasm_oof_predictions(model, dataloader):
    """
    Generate out-of-fold sarcasm probabilities for one validation fold.

    IMPORTANT:
    This function is only used on a fold's validation data.
    It must NOT be used on the final valid.csv test data for
    threshold optimization.
    """

    model.eval()

    all_probs = []
    all_labels = []
    all_varieties = []

    for batch in dataloader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sarcasm_logits = outputs["sarcasm_logits"]

        sarcasm_probs = torch.softmax(
            sarcasm_logits,
            dim=1
        )[:, 1]

        all_probs.append(
            sarcasm_probs.detach().cpu().numpy()
        )

        all_labels.append(
            batch["sarcasm_labels"].cpu().numpy()
        )

        all_varieties.append(
            np.asarray(batch["variety"])
        )

    return (
        np.concatenate(all_varieties),
        np.concatenate(all_labels),
        np.concatenate(all_probs)
    )

def optimize_sarcasm_thresholds_from_oof(
    varieties,
    sarcasm_labels,
    sarcasm_positive_probs
):
    """
    Find AU and UK sarcasm thresholds using ONLY OOF predictions.

    The final competition/test set is never used here.
    """

    varieties = np.asarray(varieties)
    sarcasm_labels = np.asarray(sarcasm_labels)
    sarcasm_positive_probs = np.asarray(
        sarcasm_positive_probs
    )

    au_mask = varieties == "en-AU"
    uk_mask = varieties == "en-UK"

    thresholds = np.arange(
        0.20,
        0.801,
        0.01
    )

    best_au_threshold = 0.50
    best_au_f1 = -1.0

    best_uk_threshold = 0.50
    best_uk_f1 = -1.0

    # ============================================================
    # AU
    # ============================================================

    au_labels = sarcasm_labels[au_mask]
    au_probs = sarcasm_positive_probs[au_mask]

    for threshold in thresholds:

        predictions = (
            au_probs >= threshold
        ).astype(np.int64)

        score = f1_score(
            au_labels,
            predictions,
            average="macro",
            zero_division=0
        )

        if score > best_au_f1:

            best_au_f1 = score
            best_au_threshold = float(threshold)

    # ============================================================
    # UK
    # ============================================================

    uk_labels = sarcasm_labels[uk_mask]
    uk_probs = sarcasm_positive_probs[uk_mask]

    for threshold in thresholds:

        predictions = (
            uk_probs >= threshold
        ).astype(np.int64)

        score = f1_score(
            uk_labels,
            predictions,
            average="macro",
            zero_division=0
        )

        if score > best_uk_f1:

            best_uk_f1 = score
            best_uk_threshold = float(threshold)

    official_sarcasm_score = min(
        best_au_f1,
        best_uk_f1
    )

    print("\n" + "=" * 70)
    print("OOF SARCASM THRESHOLD OPTIMIZATION")
    print("=" * 70)

    print(
        f"AU OOF threshold : {best_au_threshold:.2f}"
    )

    print(
        f"AU OOF Macro F1  : {best_au_f1:.4f}"
    )

    print(
        f"UK OOF threshold : {best_uk_threshold:.2f}"
    )

    print(
        f"UK OOF Macro F1  : {best_uk_f1:.4f}"
    )

    print(
        f"OOF official sarcasm score: "
        f"{official_sarcasm_score:.4f}"
    )

    print("=" * 70)

    return (
        best_au_threshold,
        best_uk_threshold
    )

def predict_sarcasm_with_dialect_thresholds(
    varieties,
    sarcasm_positive_probs,
    au_threshold,
    uk_threshold
):
    """
    Apply previously learned OOF thresholds to new data.

    IMPORTANT:
    The thresholds are already fixed.
    No labels from the new data are used.
    """

    varieties = np.asarray(varieties)
    sarcasm_positive_probs = np.asarray(
        sarcasm_positive_probs
    )

    predictions = np.zeros(
        len(sarcasm_positive_probs),
        dtype=np.int64
    )

    au_mask = varieties == "en-AU"
    uk_mask = varieties == "en-UK"

    predictions[au_mask] = (
        sarcasm_positive_probs[au_mask]
        >= au_threshold
    ).astype(np.int64)

    predictions[uk_mask] = (
        sarcasm_positive_probs[uk_mask]
        >= uk_threshold
    ).astype(np.int64)

    return predictions

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
    if os.path.isfile(valid_path):
        valid_df = pd.read_csv(valid_path)
        df = pd.concat([df, valid_df], ignore_index=True)


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

    if USE_ISARCASM_PRETRAINING or USE_REDDIT_PRETRAINING:
        pretrained_checkpoint = os.path.join( GENERAL_ADAPTER_CHECKPOINT, "best_general_adapter.pt", )
        if os.path.exists(
            pretrained_checkpoint
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
            if USE_ISARCASM_PRETRAINING:
                sarcasm_pretrain_general_adapter()

            if USE_REDDIT_PRETRAINING:
                pretrain_reddit_general_adapter( )
                


    # df["stratify_group"] =  df["variety"].astype(str)+ "_" + df["sarcasm"].astype(str)
    df["stratify_group"] = df["variety"].astype(str) + "_" + df["sentiment"].astype(str)+ "_" + df["sarcasm"].astype(str)

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=SEED
    )

    os.makedirs(
        ALTA_CHECKPOINT_DIR,
        exist_ok=True
    )

    fold_scores = []

    oof_varieties = []
    oof_sarcasm_labels = []
    oof_sarcasm_probs = []
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
        if USE_ISARCASM_PRETRAINING or USE_REDDIT_PRETRAINING:
            initialize_sarcasm_adapters_from_pretrained(
                model=model,
                checkpoint_path=GENERAL_ADAPTER_CHECKPOINT,
            )
            force_lora_parameters_to_float32( model )
            enable_all_lora_gradients(model)
            check_model_finite(model)
            print("\nLoRA parameter dtypes:")

            for name, param in model.named_parameters():

                if ("lora_A" in name or "lora_B" in name):
                    print(f"{name}: {param.dtype}")
        else:
            force_lora_parameters_to_float32(model)

        # print("\nAvailable adapters:")

        # print(model.encoder.peft_config.keys())
        
        

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




        
       
        trainable_parameters = [
            param
            for param in model.parameters()
            if param.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            eps=1e-6,
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
        # ============================================================
        # DEBUG: CHECK FORWARD OUTPUTS AND LOSSES BEFORE TRAINING
        # ============================================================

        model.eval()

        debug_batch = next(iter(train_loader))

        debug_input_ids = debug_batch[
            "input_ids"
        ].to(DEVICE)

        debug_attention_mask = debug_batch[
            "attention_mask"
        ].to(DEVICE)

        debug_variety_ids = debug_batch[
            "variety_ids"
        ].to(DEVICE)

        debug_sentiment_labels = debug_batch[
            "sentiment_labels"
        ].to(DEVICE)

        debug_sarcasm_labels = debug_batch[
            "sarcasm_labels"
        ].to(DEVICE)


        with torch.no_grad():

            debug_outputs = model(
                input_ids=debug_input_ids,
                attention_mask=debug_attention_mask,
                variety_ids=debug_variety_ids,
            )


        print("\n" + "=" * 70)
        print("FORWARD / LOSS DEBUG")
        print("=" * 70)

        sentiment_logits = debug_outputs[
            "sentiment_logits"
        ]

        sarcasm_logits = debug_outputs[
            "sarcasm_logits"
        ]

        print(
            "Sentiment logits finite:",
            torch.isfinite(sentiment_logits).all().item()
        )

        print(
            "Sarcasm logits finite:",
            torch.isfinite(sarcasm_logits).all().item()
        )

        print(
            "Sentiment logits:\n",
            sentiment_logits
        )

        print(
            "Sarcasm logits:\n",
            sarcasm_logits
        )


        # ------------------------------------------------------------
        # Check individual losses
        # ------------------------------------------------------------

        sentiment_debug_loss = sentiment_loss(
            sentiment_logits,
            debug_sentiment_labels,
        )

        au_mask = (
            debug_variety_ids == 0
        )

        uk_mask = (
            debug_variety_ids == 1
        )

        print(
            "\nSentiment loss:",
            sentiment_debug_loss.item()
        )

        if au_mask.any():

            au_debug_loss = au_sarcasm_loss(
                sarcasm_logits[au_mask],
                debug_sarcasm_labels[au_mask],
            )

            print(
                "AU sarcasm loss:",
                au_debug_loss.item()
            )

        else:
            au_debug_loss = None


        if uk_mask.any():

            uk_debug_loss = uk_sarcasm_loss(
                sarcasm_logits[uk_mask],
                debug_sarcasm_labels[uk_mask],
            )

            print(
                "UK sarcasm loss:",
                uk_debug_loss.item()
            )

        else:
            uk_debug_loss = None


        print("=" * 70)


        for epoch in range(NUM_EPOCHS):
            # print("\n" + "=" * 70)
            # print("FINAL LoRA DTYPE CHECK")
            # print("=" * 70)

            bad_dtype = []

            for name, param in model.named_parameters():

                if (
                    "lora_A" in name
                    or "lora_B" in name
                ):

                    if param.dtype != torch.float32:

                        bad_dtype.append(
                            (name, param.dtype)
                        )

                    # print(
                    #     f"{name}: {param.dtype}"
                    # )

            # print("=" * 70)

            if bad_dtype:

                raise RuntimeError(
                    "Some LoRA parameters are not float32."
                )

            print(
                "✓ All LoRA parameters are float32."
            )

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

                checkpoint_path = os.path.join(ALTA_CHECKPOINT_DIR,f"best_fold_{fold + 1}.pt",)

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

        checkpoint_path = os.path.join( ALTA_CHECKPOINT_DIR, f"best_fold_{fold + 1}.pt", )

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
            dropout=0.1,)

        model.to(DEVICE)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        # ============================================================
        # GENERATE OOF PREDICTIONS FOR THIS FOLD
        # ============================================================

        fold_oof_varieties, fold_oof_labels, fold_oof_probs = (
            get_sarcasm_oof_predictions(
                model=model,
                dataloader=val_loader
            )
        )

        oof_varieties.append(
            fold_oof_varieties
        )

        oof_sarcasm_labels.append(
            fold_oof_labels
        )

        oof_sarcasm_probs.append(
            fold_oof_probs
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
    # ============================================================
    # COMBINE ALL OOF PREDICTIONS
    # ============================================================

    oof_varieties = np.concatenate(
        oof_varieties
    )

    oof_sarcasm_labels = np.concatenate(
        oof_sarcasm_labels
    )

    oof_sarcasm_probs = np.concatenate(
        oof_sarcasm_probs
    )

    print("\n" + "=" * 70)
    print("OOF PREDICTIONS READY")
    print("=" * 70)

    print(
        f"OOF samples: {len(oof_sarcasm_labels)}"
    )

    print(
        f"AU samples: "
        f"{np.sum(oof_varieties == 'en-AU')}"
    )

    print(
        f"UK samples: "
        f"{np.sum(oof_varieties == 'en-UK')}"
    )
    # ============================================================
    # LEARN THRESHOLDS FROM OOF ONLY
    # ============================================================

    au_threshold, uk_threshold = (
        optimize_sarcasm_thresholds_from_oof(
            varieties=oof_varieties,

            sarcasm_labels=oof_sarcasm_labels,

            sarcasm_positive_probs=oof_sarcasm_probs
        )
    )
    # ============================================================
    # FINAL ENSEMBLE ON VALID.CSV
    # ============================================================

    ensemble_output = predict_ensemble(
        models=ensemble_models,
        dataloader=test_loader
    )


    # ============================================================
    # SENTIMENT
    # Keep normal argmax
    # ============================================================

    final_sentiment_predictions = (
        ensemble_output["sentiment_predictions"]
    )


    # ============================================================
    # SARCASM
    # Apply the thresholds learned from OOF
    # ============================================================

    final_sarcasm_predictions = (
        predict_sarcasm_with_dialect_thresholds(

            varieties=ensemble_output["varieties"],

            sarcasm_positive_probs=ensemble_output[
                "sarcasm_positive_probs"
            ],

            au_threshold=au_threshold,

            uk_threshold=uk_threshold
        )
    )
#########--------------

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

    answer_df["sentiment"] = final_sentiment_predictions
    answer_df["sarcasm"] = final_sarcasm_predictions

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