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
        self.w_1 = nn.Conv1d(d_m, d_ff, 1)
        self.w_2 = nn.Conv1d(d_ff, d_m, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_m)

    def forward(self, x):
        residual = x
        x = x.transpose(1,2)
        output = self.w_2(F.relu(self.w_1(x))) # (64, 200, 512)
        output = output.transpose(1, 2)

        output = self.dropout(output)
        output = self.layer_norm(output + residual) # residual connection
        return output