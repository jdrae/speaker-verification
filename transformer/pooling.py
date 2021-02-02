import torch
import torch.nn as nn

class SelfAttentionPooling(nn.Module):
    def __init__(self, d_m, dropout=0.1):
        super().__init__()
        self.d_m = d_m
        self.softmax = nn.Softmax(dim=2)
        self.w_c = nn.Linear(d_m, 1)
    
    def forward(self, x): # (bs, T, d_m)
        attn = self.w_c(x).transpose(1,2) # (bs, 1, T)
        attn = self.softmax(attn)
        attn = torch.bmm(attn, x) # (bs, 1, d_m)
        return attn
        
