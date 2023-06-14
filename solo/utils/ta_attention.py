import math
import torch
import torch.nn as nn


class TA_Attention(nn.Module):
    def __init__(
        self,
        input_dim,
        query_dim,
        value_dim,
        hidden_dim=None,
        num_heads=1,
        bias=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
    ):
        super().__init__()
        assert value_dim % num_heads == 0, "value dim must be divisible by num_heads"
        assert query_dim % num_heads == 0, "query dim must be divisible by num_heads"
        self.value_dim = value_dim
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.value_head_dim = value_dim // num_heads
        self.query_head_dim = query_dim // num_heads
        self.d = math.sqrt(self.query_head_dim)

        # Input dim?
        self.qkv_transform = (
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=bias),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, query_dim * 2 + value_dim, bias=bias),
                nn.BatchNorm1d(query_dim * 2 + value_dim),
                nn.ReLU(),
            )
            if hidden_dim
            else 
                nn.Linear(input_dim, query_dim * 2 + value_dim, bias=bias)
            
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Sequential(
                nn.Linear(value_dim, value_dim),
                nn.BatchNorm1d(value_dim),
                nn.ReLU(),
            )
        self.proj_dropout = nn.Dropout(proj_dropout)

    # X has shape [B, D]
    def forward(self, x):
        batch_size, _ = x.shape
        qkv = self.qkv_transform(x)
        qv = qkv[:, : self.query_dim * 2]
        q, k = qv.reshape(batch_size, 2, self.num_heads, self.query_head_dim).permute(1, 2, 0, 3).unbind(dim=0)
        v = qkv[:, self.query_dim * 2 :].reshape(batch_size, self.num_heads, self.value_head_dim).permute(1, 0, 2)

        return q, k, v

    def attention(self, q, k, v):
        _, batch_size, _ = q.shape
        attn_weights = torch.matmul(q, k.transpose(-2, -1)).contiguous() / self.d
        attn_weights = attn_weights.softmax(dim=-1)

        with torch.no_grad():
            # Zero out the 5 highest sum columns of each head
            indices = torch.topk(attn_weights.sum(dim=1), k=5, dim=-1).indices
            for idx, t in enumerate(indices):
                attn_weights[idx, :, t] = 0

        out = torch.matmul(self.attn_dropout(attn_weights), v)
        out = out.transpose(0, 1).reshape(batch_size, self.value_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)

        # out = out + self.ffn(out)

        return out, attn_weights
