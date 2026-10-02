import torch
import torch.nn as nn

from transformers import AutoModel
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)


import torch
import torch.nn as nn

from transformers import AutoModel

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)


class TaskClassificationHead(nn.Module):

    def __init__(
        self,
        hidden_size,
        dropout=0.1,
        num_labels=2,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
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
        dropout=0.1,
    ):
        super().__init__()

        self.model_name = model_name

        # ============================================================
        # Base DeBERTa
        # ============================================================

        base_model = AutoModel.from_pretrained(
            model_name
        )

        hidden_size = (
            base_model.config.hidden_size
        )

        # ============================================================
        # LoRA configuration
        # ============================================================

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

        # ============================================================
        # Create first adapter
        # ============================================================

        self.encoder = get_peft_model(
            base_model,
            lora_config,
            adapter_name="sentiment_au",
        )

        # ============================================================
        # SENTIMENT adapters
        # ============================================================

        self.encoder.add_adapter(
            "sentiment_uk",
            lora_config,
        )

        # ============================================================
        # SARCASM adapters
        # ============================================================

        self.encoder.add_adapter(
            "sarcasm_au",
            lora_config,
        )

        self.encoder.add_adapter(
            "sarcasm_uk",
            lora_config,
        )

        # ============================================================
        # Task-specific heads
        # ============================================================

        self.sentiment_head = TaskClassificationHead(
            hidden_size=hidden_size,
            dropout=dropout,
            num_labels=2,
        )

        self.sarcasm_head = TaskClassificationHead(
            hidden_size=hidden_size,
            dropout=dropout,
            num_labels=2,
        )

        print("\n" + "=" * 70)
        print("TASK-SPECIFIC LoRA ARCHITECTURE")
        print("=" * 70)

        print(
            "Sentiment AU adapter : sentiment_au"
        )

        print(
            "Sentiment UK adapter : sentiment_uk"
        )

        print(
            "Sarcasm AU adapter   : sarcasm_au"
        )

        print(
            "Sarcasm UK adapter   : sarcasm_uk"
        )

        print("=" * 70)

    # ================================================================
    # Mean pooling
    # ================================================================

    def mean_pooling(
        self,
        hidden_states,
        attention_mask,
    ):
        """
        Mean pooling over non-padding tokens.
        """

        mask = attention_mask.unsqueeze(-1).float()

        summed = torch.sum(
            hidden_states * mask,
            dim=1,
        )

        counts = torch.clamp(
            mask.sum(dim=1),
            min=1e-9,
        )

        return summed / counts

    # ================================================================
    # Forward one subset through a selected adapter
    # ================================================================
    def _enable_all_lora_gradients(self):
        """
        PEFT's set_adapter() may disable gradients for inactive adapters.

        We want only ONE adapter to be active for each forward branch,
        but ALL four LoRA adapters must remain trainable so gradients
        from the complete forward pass can accumulate correctly.
        """

        adapters = [
            "sentiment_au",
            "sentiment_uk",
            "sarcasm_au",
            "sarcasm_uk",
        ]

        for name, param in self.encoder.named_parameters():

            if any(
                f"lora_A.{adapter}." in name
                or f"lora_B.{adapter}." in name
                for adapter in adapters
            ):
                param.requires_grad_(True)

    def _forward_subset(
        self,
        input_ids,
        attention_mask,
        adapter_name,
    ):
 
        self.encoder.set_adapter(
            adapter_name
        )

        # --------------------------------------------------
        # PEFT set_adapter() may disable gradients for the
        # other adapters. Re-enable all LoRA parameters.
        # --------------------------------------------------
        self._enable_all_lora_gradients()

        # --------------------------------------------------
        # Forward through the selected adapter
        # --------------------------------------------------
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # --------------------------------------------------
        # Mean pooling
        # --------------------------------------------------
        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask,
        )

        return pooled
    def forward(
        self,
        input_ids,
        attention_mask,
        variety_ids,
    ):
        """
        Four independent encoder pathways:

            AU sentiment -> sentiment_au
            UK sentiment -> sentiment_uk

            AU sarcasm   -> sarcasm_au
            UK sarcasm   -> sarcasm_uk
        """

        batch_size = input_ids.size(0)

        device = input_ids.device

        hidden_size = (
            self.encoder.base_model.config.hidden_size
        )

        # ============================================================
        # Output buffers
        # ============================================================

        sentiment_features = torch.zeros(
            batch_size,
            hidden_size,
            device=device,
        )

        sarcasm_features = torch.zeros(
            batch_size,
            hidden_size,
            device=device,
        )

        # ============================================================
        # Dialect masks
        # ============================================================

        au_indices = (
            variety_ids == 0
        ).nonzero(
            as_tuple=True
        )[0]

        uk_indices = (
            variety_ids == 1
        ).nonzero(
            as_tuple=True
        )[0]

        # ============================================================
        # SENTIMENT: en-AU
        # ============================================================

        if len(au_indices) > 0:

            au_input_ids = input_ids[
                au_indices
            ]

            au_attention_mask = attention_mask[
                au_indices
            ]

            au_sentiment_features = (
                self._forward_subset(
                    input_ids=au_input_ids,
                    attention_mask=au_attention_mask,
                    adapter_name="sentiment_au",
                )
            )

            sentiment_features[
                au_indices
            ] = au_sentiment_features

        # ============================================================
        # SENTIMENT: en-UK
        # ============================================================

        if len(uk_indices) > 0:

            uk_input_ids = input_ids[
                uk_indices
            ]

            uk_attention_mask = attention_mask[
                uk_indices
            ]

            uk_sentiment_features = (
                self._forward_subset(
                    input_ids=uk_input_ids,
                    attention_mask=uk_attention_mask,
                    adapter_name="sentiment_uk",
                )
            )

            sentiment_features[
                uk_indices
            ] = uk_sentiment_features

        # ============================================================
        # SARCASM: en-AU
        # ============================================================

        if len(au_indices) > 0:

            au_input_ids = input_ids[
                au_indices
            ]

            au_attention_mask = attention_mask[
                au_indices
            ]

            au_sarcasm_features = (
                self._forward_subset(
                    input_ids=au_input_ids,
                    attention_mask=au_attention_mask,
                    adapter_name="sarcasm_au",
                )
            )

            sarcasm_features[
                au_indices
            ] = au_sarcasm_features

        # ============================================================
        # SARCASM: en-UK
        # ============================================================

        if len(uk_indices) > 0:

            uk_input_ids = input_ids[
                uk_indices
            ]

            uk_attention_mask = attention_mask[
                uk_indices
            ]

            uk_sarcasm_features = (
                self._forward_subset(
                    input_ids=uk_input_ids,
                    attention_mask=uk_attention_mask,
                    adapter_name="sarcasm_uk",
                )
            )

            sarcasm_features[
                uk_indices
            ] = uk_sarcasm_features

        # ============================================================
        # Independent task heads
        # ============================================================

        sentiment_logits = self.sentiment_head(
            sentiment_features
        )

        sarcasm_logits = self.sarcasm_head(
            sarcasm_features
        )

        return {
            "sentiment_logits": sentiment_logits,
            "sarcasm_logits": sarcasm_logits,
        }