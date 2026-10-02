import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model

class RephraseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        tweet_embeddings,
        rephrase_embeddings,
    ):

        if tweet_embeddings.size(0) < 2:
            return torch.tensor(
                0.0,
                device=tweet_embeddings.device,
                requires_grad=True,
            )

        logits = (
            tweet_embeddings
            @ rephrase_embeddings.transpose(0, 1)
        ) / self.temperature

        labels = torch.arange(
            logits.size(0),
            device=logits.device,
        )

        loss_tweet_to_rephrase = F.cross_entropy(
            logits,
            labels,
        )

        loss_rephrase_to_tweet = F.cross_entropy(
            logits.transpose(0, 1),
            labels,
        )

        return (
            loss_tweet_to_rephrase
            + loss_rephrase_to_tweet
        ) / 2.0