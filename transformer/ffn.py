import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """Implements position-wise feedforward sublayer.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """

    def __init__(self, d_m, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_m, d_ff)
        self.w_2 = nn.Linear(d_ff, d_m)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_m)

    def forward(self, x):
        residual = x
        output = self.w_2(F.relu(self.w_1(x))) # (64, 200, 512)
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # residual connection
        return output