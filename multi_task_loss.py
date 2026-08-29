import torch.nn as nn

class MultiTaskLoss(nn.Module):

    def __init__(
        self,
        sentiment_loss,
        sarcasm_loss,
        sentiment_weight=0.4,
        sarcasm_weight=0.6
    ):

        super().__init__()

        self.sentiment_loss = (
            sentiment_loss
        )

        self.sarcasm_loss = (
            sarcasm_loss
        )

        self.sentiment_weight = (
            sentiment_weight
        )

        self.sarcasm_weight = (
            sarcasm_weight
        )


    def forward(
        self,
        sentiment_logits,
        sentiment_labels,
        sarcasm_logits,
        sarcasm_labels
    ):

        sentiment_loss = (
            self.sentiment_loss(
                sentiment_logits,
                sentiment_labels
            )
        )

        sarcasm_loss = (
            self.sarcasm_loss(
                sarcasm_logits,
                sarcasm_labels
            )
        )

        loss = (
            self.sentiment_weight
            * sentiment_loss
            +
            self.sarcasm_weight
            * sarcasm_loss
        )

        return {
            "loss": loss,
            "sentiment_loss": sentiment_loss,
            "sarcasm_loss": sarcasm_loss
        }