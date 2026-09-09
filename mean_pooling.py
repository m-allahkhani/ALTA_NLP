

import torch.nn as nn
import torch

class MeanPooling(nn.Module):
    def forward(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()

        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-6)

        return summed / counts