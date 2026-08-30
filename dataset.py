import torch

from torch.utils.data import Dataset


class ALTAMultiTaskDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=256
    ):

        self.dataframe = (
            dataframe.reset_index(drop=True)
        )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.variety_to_id = {
            "en-AU": 0,
            "en-UK": 1
        }

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.dataframe.iloc[idx]

        text = str(
            row["text"]
        )

        variety = str(
            row["variety"]
        )

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {

            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "sentiment_labels":
                torch.tensor(
                    int(row["sentiment"]),
                    dtype=torch.long
                ),

            "sarcasm_labels":
                torch.tensor(
                    int(row["sarcasm"]),
                    dtype=torch.long
                ),

            "variety_ids":
                torch.tensor(
                    self.variety_to_id[variety],
                    dtype=torch.long
                ),

            # Keep original dialect
            "variety":
                variety
        }

        return item