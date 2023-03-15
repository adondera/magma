import math
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.d = math.sqrt(self.head_dim)
        
        #Input dim?
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
    
    # X has shape [B, D]
    def forward(self, x):
        batch_size, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)
        q, k, v = torch.unbind(qkv, dim=0)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(0, 1).reshape(batch_size, self.num_heads * self.head_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out

# def scaled_dot_product_attention(q, k, v, mask):
#     """Calculate the attention weights.
#     The mask has different shapes depending on its type(padding or look ahead) 
#     but it must be broadcastable for addition.
    
#     Args:
#       q: query shape == (num_heads, batch_size, dim)
#       k: key shape == (num_heads, batch_size, dim)
#       v: value shape == (num_heads, batch_size, dim)
#       mask: Float tensor with shape broadcastable 
#             to (..., seq_len_q, seq_len_k). Defaults to None.
    
#     Returns:
#       output, attention_weights
#     """
#     num_heads, batch_size, dim = q.shape
#     attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(num_heads)
#     attn_weights = attn_weights.softmax(dim=-1)
    
#     # scale matmul_qk
#     dk = torch.tensor(k.shape[-1], dtype=torch.float32)
#     scaled_attention_logits = matmul_qk / torch.sqrt(dk)
    
#     # add the mask to the scaled tensor.
#     if mask is not None:
#         scaled_attention_logits += (mask * -1e9)  
    
#     # softmax is normalized on the last axis (seq_len_k) so that the scores
#     # add up to 1.
#     attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)
    
#     output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
#     return output, attention_weights


# class TA(nn.Module):
#     def __init__(self, dim, num_heads=1, bias=False, attn_dropout=0., proj_dropout=0.):
#         super().__init__()
#         assert dim % num_heads == 0, "dim must be divisible by num_heads"
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.d = math.sqrt(self.head_dim)
        
#         #Input dim?
#         self.qkv = nn.Linear(dim, dim * 3, bias=bias)
#         self.attn_dropout = nn.Dropout(attn_dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_dropout = nn.Dropout(proj_dropout)
    
#     def forward(self, branch1_inputs, branch2_inputs):
#         out = self.attention(x)
#         out = self.proj(out)
#         out = self.proj_dropout(out)
#         return out
