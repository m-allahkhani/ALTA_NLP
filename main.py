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
train_path = f"/kaggle//input//datasets//maryamallahkhani//official-alta-dataset//train.csv" 
test_path = f"/kaggle//input//datasets//maryamallahkhani//official-alta-dataset//valid.csv" 

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

# NUM_FOLDS = 2

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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



def train_one_epoch(model,dataloader,optimizer,scheduler,loss_function):

    model.train()

    total_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(
        dataloader
    ):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        variety_ids = batch["variety_ids"].to(DEVICE)

        sentiment_labels = batch["sentiment_labels"].to(DEVICE)

        sarcasm_labels = batch["sarcasm_labels"].to(DEVICE)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask,variety_ids=variety_ids)

        losses = loss_function(
            sentiment_logits= outputs["sentiment_logits"],
            sentiment_labels= sentiment_labels,
            sarcasm_logits= outputs["sarcasm_logits"],
            sarcasm_labels= sarcasm_labels)

        loss = losses["loss"]/GRADIENT_ACCUMULATION_STEPS
        

        loss.backward()

        if ((step + 1) % GRADIENT_ACCUMULATION_STEPS== 0):

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad()

        total_loss += (loss.item() *GRADIENT_ACCUMULATION_STEPS)

    

    if (len(dataloader) % GRADIENT_ACCUMULATION_STEPS != 0):

        torch.nn.utils.clip_grad_norm_( model.parameters(),  1.0)

        optimizer.step()

        scheduler.step()

        optimizer.zero_grad()

    return (total_loss / len(dataloader))


@torch.no_grad()
def evaluate(model,dataloader,loss_function):

    model.eval()

    total_loss = 0.0

    sentiment_labels_all = []
    sentiment_predictions_all = []

    sarcasm_labels_all = []
    sarcasm_predictions_all = []

    for batch in dataloader:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        variety_ids = batch["variety_ids"].to(DEVICE)

        sentiment_labels = batch[
            "sentiment_labels"
        ].to(DEVICE)

        sarcasm_labels = batch["sarcasm_labels"].to(DEVICE)

        outputs = model(input_ids=input_ids,attention_mask=attention_mask,variety_ids=variety_ids)

        losses = loss_function(
            sentiment_logits=outputs["sentiment_logits"],
            sentiment_labels=sentiment_labels,
            sarcasm_logits=outputs["sarcasm_logits"],
            sarcasm_labels=sarcasm_labels)

        total_loss += (losses["loss"].item())

        sentiment_predictions = torch.argmax(outputs["sentiment_logits"],dim=1)

        sarcasm_predictions = torch.argmax(outputs["sarcasm_logits"],dim=1)

        sentiment_labels_all.extend(sentiment_labels.cpu().numpy())

        sentiment_predictions_all.extend(sentiment_predictions.cpu().numpy())

        sarcasm_labels_all.extend(sarcasm_labels.cpu().numpy())

        sarcasm_predictions_all.extend(sarcasm_predictions.cpu().numpy())

    sentiment_metrics = calculate_metrics(sentiment_labels_all,sentiment_predictions_all)
    
    sarcasm_metrics = calculate_metrics(sarcasm_labels_all,sarcasm_predictions_all)

    return {

        "loss": (
            total_loss
            /
            len(dataloader)
        ),

        "sentiment":
            sentiment_metrics,

        "sarcasm":
            sarcasm_metrics
    }


@torch.no_grad()
def predict_ensemble(
    models,
    dataloader
):
    """
    Run all fold models and average their logits.

    Returns:
        varieties
        sentiment_labels
        sentiment_predictions
        sarcasm_labels
        sarcasm_predictions
    """

    for model in models:
        model.eval()

    varieties = []

    sentiment_labels_all = []
    sarcasm_labels_all = []

    sentiment_logits_all_models = []
    sarcasm_logits_all_models = []

    # ----------------------------------------------
    # Run every model
    # ----------------------------------------------

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

        # These are identical for every model.
        if model_index == 0:

            sentiment_labels_all = (
                sentiment_labels_model
            )

            sarcasm_labels_all = (
                sarcasm_labels_model
            )

            varieties = varieties_model

    # ==================================================
    # Average logits
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
    # Final predictions
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

   

    df["stratify_group"] = (
        df["variety"].astype(str)+ "_" + df["sarcasm"].astype(str))

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

       
        model = (
            DialectAwareMultiTaskDeBERTa(
                model_name=MODEL_NAME,
                lora_r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                dropout=0.1
            )
        )

        model.to(DEVICE)

        
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

        
        steps_per_epoch = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS)

     
        steps_per_epoch = max(1, steps_per_epoch)
        
        total_training_steps = (steps_per_epoch * NUM_EPOCHS)

        warmup_steps = int( total_training_steps *  WARMUP_RATIO)

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps= warmup_steps,
                num_training_steps=total_training_steps)
        


        best_score = -1.0

        for epoch in range(NUM_EPOCHS):

            print( f"\nEpoch {epoch + 1} / {NUM_EPOCHS}")

            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_function=loss_function
            )

            results = evaluate(
                model=model,
                dataloader=val_loader,
                loss_function=loss_function
            )

            print(f"Train Loss: {train_loss:.4f}")

            print(f"Validation Loss: {results['loss']:.4f}")

            print(
                "Sentiment Macro F1: "
                f"{results['sentiment']['macro_f1']:.4f}")

            print(
                "Sarcasm Macro F1: "
                f"{results['sarcasm']['macro_f1']:.4f}")

            

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

            if combined_score > best_score:

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
            f"\nBest Fold Score: "
            f"{best_score:.4f}"
        )

        # ----------------------------------------------
        # Free fold model
        # ----------------------------------------------

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

    # ==================================================
    # Print official metrics
    # ==================================================

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

    print(
        "=" * 60
    )

    print(
        "\nFirst 5 predictions:"
    )

    print(
        answer_df.head()
    )


    

if __name__ == "__main__":
    args = parse_args()
    main(args)