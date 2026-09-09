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
    get_peft_model
)

import torch.nn.functional as F
from mean_pooling import MeanPooling
class GeneralSarcasmPretrainModel(nn.Module):
    """
    General sarcasm adaptation model.

    Objectives:
        1. Binary sarcasm classification:
           tweet -> sarcastic

        2. Rephrase contrastive learning:
           tweet <-> non-sarcastic rephrase

        3. Fine-grained sarcasm category:
           tweet -> sarcasm category
    """

    def __init__(
        self,
        model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        dropout=0.1,
    ):
        super().__init__()

        base_model = AutoModel.from_pretrained(model_name)

        hidden_size = base_model.config.hidden_size

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[
                "query_proj",
                "key_proj",
                "value_proj",
            ],
        )

        self.encoder = get_peft_model(
            base_model,
            lora_config,
            adapter_name="general",
        )

        self.pooler = MeanPooling()

        # Main binary sarcasm head
        self.binary_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

        # Fine-grained "sarcasm" category head
        self.fine_sarcasm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Projection used only for contrastive learning
        projection_dim = 256

        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            adapter_name="general",
        )

        pooled = self.pooler(
            outputs.last_hidden_state,
            attention_mask,
        )

        return pooled

    def forward(
        self,
        tweet_input_ids,
        tweet_attention_mask,
        rephrase_input_ids=None,
        rephrase_attention_mask=None,
    ):
        tweet_repr = self.encode(
            tweet_input_ids,
            tweet_attention_mask,
        )

        binary_logits = self.binary_head(tweet_repr)

        fine_logits = self.fine_sarcasm_head(
            tweet_repr
        ).squeeze(-1)

        output = {
            "tweet_repr": tweet_repr,
            "binary_logits": binary_logits,
            "fine_logits": fine_logits,
        }

        if (
            rephrase_input_ids is not None
            and rephrase_attention_mask is not None
        ):
            rephrase_repr = self.encode(
                rephrase_input_ids,
                rephrase_attention_mask,
            )

            tweet_proj = F.normalize(
                self.contrastive_projection(tweet_repr),
                dim=-1,
            )

            rephrase_proj = F.normalize(
                self.contrastive_projection(rephrase_repr),
                dim=-1,
            )

            output["tweet_proj"] = tweet_proj
            output["rephrase_proj"] = rephrase_proj

        return output