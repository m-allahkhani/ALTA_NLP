import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):

        super().__init__()

        self.gamma = gamma

        if alpha is not None:

            self.register_buffer(
                "alpha",
                
                alpha.detach().clone().float()
                
            )

        else:

            self.alpha = None


    def forward(
        self,
        logits,
        targets
    ):

        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none"
        )

        pt = torch.exp(
            -ce_loss
        )

        loss = (
            (1 - pt)
            ** self.gamma
        ) * ce_loss

        if self.alpha is not None:

            alpha_t = self.alpha[
                targets
            ]

            loss = (
                alpha_t * loss
            )

        return loss.mean()

