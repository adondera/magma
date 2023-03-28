import math
import torch
import torch.nn as nn


class TA_Attention(nn.Module):
    def __init__(
        self, dim, num_heads=1, bias=False, attn_dropout=0.0, proj_dropout=0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.d = math.sqrt(self.head_dim)

        # Input dim?
        self.qkv_transform = nn.Sequential(
            nn.Linear(dim, dim // 2, bias=bias),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim * 3, bias=bias),
            nn.BatchNorm1d(dim * 3),
            nn.ReLU()
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU())
        self.proj_dropout = nn.Dropout(proj_dropout)
        # self.ffn = nn.Sequential(
        #     nn.Linear(dim, dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(dim // 2, dim),
        #     nn.BatchNorm1d(dim),
        #     nn.ReLU(),
        # )

    # X has shape [B, D]
    def forward(self, x):
        batch_size, _ = x.shape
        qkv = self.qkv_transform(x)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = torch.unbind(qkv, dim=0)

        return q, k, v

    def attention(self, q, k, v):
        _, batch_size, _ = q.shape
        attn_weights = torch.matmul(q, k.transpose(-2, -1)).contiguous() / math.sqrt(self.head_dim)
        attn_weights = attn_weights.softmax(dim=-1)

        out = torch.matmul(self.attn_dropout(attn_weights), v)
        out = out.transpose(0, 1).reshape(batch_size, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)

        # out = out + self.ffn(out)

        return out, attn_weights
