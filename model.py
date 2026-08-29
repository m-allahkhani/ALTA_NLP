import torch
import torch.nn as nn

from transformers import AutoModel
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)


class TaskClassificationHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        dropout=0.1,
        num_labels=2
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


class DialectAwareMultiTaskDeBERTa(nn.Module):
    def __init__(
        self,
        model_name="microsoft/deberta-v3-base",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        dropout=0.1
    ):
        super().__init__()

        self.model_name = model_name

        base_model = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = base_model.config.hidden_size

        # -------------------------------------------------
        # LoRA configuration
        # -------------------------------------------------

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,

            r=lora_r,

            lora_alpha=lora_alpha,

            lora_dropout=lora_dropout,

            bias="none",

            target_modules=[
                "query_proj",
                "key_proj",
                "value_proj"
            ]
        )

        # -------------------------------------------------
        # Create General LoRA adapter
        # -------------------------------------------------

        self.encoder = get_peft_model(
            base_model,
            lora_config,
            adapter_name="general"
        )

        # -------------------------------------------------
        # Add AU adapter
        # -------------------------------------------------

        self.encoder.add_adapter(
            "au",
            lora_config
        )

        # -------------------------------------------------
        # Add UK adapter
        # -------------------------------------------------

        self.encoder.add_adapter(
            "uk",
            lora_config
        )

        # -------------------------------------------------
        # Create combined adapters
        #
        # general_au = general + AU
        # general_uk = general + UK
        # -------------------------------------------------

        self.encoder.add_weighted_adapter(
            adapters=[
                "general",
                "au"
            ],
            weights=[
                1.0,
                1.0
            ],
            adapter_name="general_au",
            combination_type="linear"
        )

        self.encoder.add_weighted_adapter(
            adapters=[
                "general",
                "uk"
            ],
            weights=[
                1.0,
                1.0
            ],
            adapter_name="general_uk",
            combination_type="linear"
        )

        # -------------------------------------------------
        # Task-specific heads
        # -------------------------------------------------

        self.sentiment_head = TaskClassificationHead(
            hidden_size=hidden_size,
            dropout=dropout,
            num_labels=2
        )

        self.sarcasm_head = TaskClassificationHead(
            hidden_size=hidden_size,
            dropout=dropout,
            num_labels=2
        )

    def mean_pooling(
        self,
        hidden_states,
        attention_mask
    ):
        """
        Mean pooling over non-padding tokens.
        """

        mask = attention_mask.unsqueeze(-1).float()

        summed = torch.sum(
            hidden_states * mask,
            dim=1
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        return summed / counts

    def _forward_subset(
        self,
        input_ids,
        attention_mask,
        adapter_name
    ):
        """
        Forward one dialect-specific subset.
        """

        self.encoder.set_adapter(
            adapter_name
        )

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        return pooled

    def forward(
        self,
        input_ids,
        attention_mask,
        variety_ids
    ):

        batch_size = input_ids.size(0)

        device = input_ids.device

        hidden_size = (
            self.encoder.base_model.config.hidden_size
        )

        pooled_output = torch.zeros(
            batch_size,
            hidden_size,
            device=device
        )

        # ---------------------------------------------
        # AU examples
        #
        # variety_ids:
        # 0 = en-AU
        # 1 = en-UK
        # ---------------------------------------------

        au_indices = (
            variety_ids == 0
        ).nonzero(
            as_tuple=True
        )[0]

        if len(au_indices) > 0:

            au_input_ids = input_ids[
                au_indices
            ]

            au_attention_mask = attention_mask[
                au_indices
            ]

            au_features = self._forward_subset(
                input_ids=au_input_ids,
                attention_mask=au_attention_mask,
                adapter_name="general_au"
            )

            pooled_output[
                au_indices
            ] = au_features

        # ---------------------------------------------
        # UK examples
        # ---------------------------------------------

        uk_indices = (
            variety_ids == 1
        ).nonzero(
            as_tuple=True
        )[0]

        if len(uk_indices) > 0:

            uk_input_ids = input_ids[
                uk_indices
            ]

            uk_attention_mask = attention_mask[
                uk_indices
            ]

            uk_features = self._forward_subset(
                input_ids=uk_input_ids,
                attention_mask=uk_attention_mask,
                adapter_name="general_uk"
            )

            pooled_output[
                uk_indices
            ] = uk_features

        # ---------------------------------------------
        # Task-specific predictions
        # ---------------------------------------------

        sentiment_logits = self.sentiment_head(
            pooled_output
        )

        sarcasm_logits = self.sarcasm_head(
            pooled_output
        )

        return {
            "sentiment_logits": sentiment_logits,
            "sarcasm_logits": sarcasm_logits
        }