import os
import random

import numpy as np
import pandas as pd
import math
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
    calculate_metrics,
    evaluate_final_test
)

from focal_loss import FocalLoss
from multi_task_loss import MultiTaskLoss

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

MAX_LENGTH = 128

BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 4

# NUM_EPOCHS = 1

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.1

EARLY_STOPPING_PATIENCE = 2 
MIN_DELTA = 0.001

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
    loss_function
):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        variety_ids = batch["variety_ids"].to(DEVICE)

        sentiment_labels = (
            batch["sentiment_labels"].to(DEVICE)
        )

        sarcasm_labels = (
            batch["sarcasm_labels"].to(DEVICE)
        )

 

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            variety_ids=variety_ids
        )

     

        losses = loss_function(
            sentiment_logits=outputs["sentiment_logits"],
            sentiment_labels=sentiment_labels,
            sarcasm_logits=outputs["sarcasm_logits"],
            sarcasm_labels=sarcasm_labels
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
            == 0
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

    return total_loss / len(dataloader)



@torch.no_grad()
def evaluate(
    model,
    dataloader,
    loss_function
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

        # ----------------------------------------------
        # Loss
        # ----------------------------------------------

        losses = loss_function(
            sentiment_logits=outputs["sentiment_logits"],
            sentiment_labels=sentiment_labels,
            sarcasm_logits=outputs["sarcasm_logits"],
            sarcasm_labels=sarcasm_labels
        )

        total_loss += losses["loss"].item()

        # ----------------------------------------------
        # Predictions
        # ----------------------------------------------

        sentiment_predictions = torch.argmax(
            outputs["sentiment_logits"],
            dim=1
        )

        sarcasm_predictions = torch.argmax(
            outputs["sarcasm_logits"],
            dim=1
        )

        # Move to CPU / NumPy
        variety_ids_cpu = variety_ids.cpu().numpy()

        sentiment_labels_cpu = sentiment_labels.cpu().numpy()
        sentiment_predictions_cpu = sentiment_predictions.cpu().numpy()

        sarcasm_labels_cpu = sarcasm_labels.cpu().numpy()
        sarcasm_predictions_cpu = sarcasm_predictions.cpu().numpy()

        # --------------------------------------------------
        # Separate AU and UK samples
        #
        # IMPORTANT:
        # This assumes:
        #   variety_id == 0 -> en-AU
        #   variety_id == 1 -> en-UK
        #
        # If your dataset.py uses the opposite mapping,
        # swap these two conditions.
        # --------------------------------------------------

        for i in range(len(variety_ids_cpu)):

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

    # --------------------------------------------------
    # Extract Macro F1
    # --------------------------------------------------

    sentiment_en_au = sentiment_au_metrics["macro_f1"]
    sentiment_en_uk = sentiment_uk_metrics["macro_f1"]

    sarcasm_en_au = sarcasm_au_metrics["macro_f1"]
    sarcasm_en_uk = sarcasm_uk_metrics["macro_f1"]

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
    # Return everything
    # ==================================================

    return {

        "loss": (
            total_loss /
            len(dataloader)
        ),

        "sentiment": {

            "en-AU": sentiment_en_au,

            "en-UK": sentiment_en_uk,

            # Pooled Macro F1, useful for monitoring
            # but NOT used for checkpoint selection.
            "macro_f1": (
                sentiment_en_au +
                sentiment_en_uk
            ) / 2.0
        },

        "sarcasm": {

            "en-AU": sarcasm_en_au,

            "en-UK": sarcasm_en_uk,

            # Pooled-by-dialect average, useful for monitoring
            # but NOT used for checkpoint selection.
            "macro_f1": (
                sarcasm_en_au +
                sarcasm_en_uk
            ) / 2.0
        },

        # These are the actual task scores used
        # by the ALTA competition.
        "sentiment_score": sentiment_score,

        "sarcasm_score": sarcasm_score,

        # This MUST be used for best-checkpoint selection.
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

        model.to(DEVICE)

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

        
        sentiment_weights =  compute_class_weights( train_df["sentiment"] ).to(DEVICE)

        sarcasm_weights = compute_class_weights( train_df["sarcasm"]).to(DEVICE) 

        sentiment_loss = nn.CrossEntropyLoss(weight=sentiment_weights )

        sarcasm_loss = FocalLoss(alpha=sarcasm_weights,gamma=2.0)

        loss_function =  MultiTaskLoss(
                sentiment_loss=sentiment_loss,
                sarcasm_loss=sarcasm_loss,
                sentiment_weight=0.4,
                sarcasm_weight=0.6
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
                loss_function=loss_function
            )

            # ----------------------------------------------
            # Validation
            # ----------------------------------------------

            results = evaluate(
                model=model,
                dataloader=val_loader,
                loss_function=loss_function
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