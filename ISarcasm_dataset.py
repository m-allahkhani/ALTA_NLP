from torch.utils.data import (
    DataLoader,
    Dataset,
    random_split
)
import torch
import pandas as pd
import numpy as np
# ============================================================
# iSarcasmEval MULTI-TASK DATASET
# ============================================================

class ISarcasmDataset(Dataset):
    """
    Uses:
        tweet       -> binary sarcastic supervision
        sarcastic   -> binary sarcastic label
        rephrase     -> contrastive pair with tweet
        sarcasm      -> fine-grained sarcasm-category supervision

    The 'sarcasm' category is only used when it is actually annotated.
    Missing values are masked rather than automatically treated as 0.
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_length=128,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Main binary label
        self.sarcastic_label = pd.to_numeric(
            self.df["sarcastic"],
            errors="coerce"
        ).fillna(0).astype(int).values

        # Fine-grained "sarcasm" category.
        # Keep missing values as -1 so they can be masked during loss.
        fine = pd.to_numeric(
            self.df["sarcasm"],
            errors="coerce"
        )

        self.fine_sarcasm_label = np.where(
            fine.notna(),
            fine.fillna(0).astype(int),
            -1
        )

        # Rephrase availability
        if "rephrase" in self.df.columns:
            rephrase = self.df["rephrase"].fillna("").astype(str).str.strip()
            self.rephrase = rephrase.values
        else:
            self.rephrase = np.array([""] * len(self.df))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        tweet = str(row["tweet"])
        rephrase = self.rephrase[idx]

        tweet_enc = self.tokenizer(
            tweet,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "tweet_input_ids": tweet_enc["input_ids"].squeeze(0),
            "tweet_attention_mask": tweet_enc["attention_mask"].squeeze(0),

            "sarcastic_label": torch.tensor(
                self.sarcastic_label[idx],
                dtype=torch.long
            ),

            "fine_sarcasm_label": torch.tensor(
                self.fine_sarcasm_label[idx],
                dtype=torch.long
            ),

            # 1 if a valid rephrase exists
            "has_rephrase": torch.tensor(
                1 if rephrase else 0,
                dtype=torch.long
            ),
        }

        # Only tokenize rephrase if present.
        if rephrase:
            rep_enc = self.tokenizer(
                rephrase,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            item["rephrase_input_ids"] = rep_enc["input_ids"].squeeze(0)
            item["rephrase_attention_mask"] = rep_enc[
                "attention_mask"
            ].squeeze(0)
        else:
            item["rephrase_input_ids"] = torch.zeros(
                self.max_length,
                dtype=torch.long
            )
            item["rephrase_attention_mask"] = torch.zeros(
                self.max_length,
                dtype=torch.long
            )

        return item