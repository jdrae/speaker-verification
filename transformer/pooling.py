import torch
import torch.nn as nn

class SelfAttentionPooling(nn.Module):
    def __init__(self, d_m, dropout=0.1):
        super().__init__()
        self.d_m = d_m
        self.softmax = nn.Softmax(dim=2)
        self.w_c = nn.Linear(d_m, 1)
    
    def forward(self, x): # (64, 200, 512)
        attn = self.w_c(x).transpose(1,2) # (64, 1, 200)
        attn = self.softmax(attn)
        attn = torch.bmm(attn, x) # (64, 1, 512)
        return attn
        
