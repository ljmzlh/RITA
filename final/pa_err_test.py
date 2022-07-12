import torch
from performer_pytorch import FastAttention
import torch
import torch.nn as nn
import math


def va(q,k,v):
    b,h,n,d=q.size()
    attention_scores = torch.matmul(q, k.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(d)
    
    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    attn = attention_probs

    context_layer = torch.matmul(attn, v)

    return context_layer,attn


# queries / keys / values with heads already split and transposed to first dimension
# 8 heads, dimension of head is 64, sequence length of 512
b,h,n,d=1,8,1024,16

q = torch.randn(b,h,n,d)
k = torch.randn(b,h,n,d)
v = torch.randn(b,h,n,d)

attn_fn = FastAttention(dim_heads = d , nb_features=None, causal = False)

pe_out,qk = attn_fn(q, k, v) # (1, 8, 512, 64)
va_out,a = va(q,k,v)



error=torch.abs(qk-a)
print(error.mean())
print(error.max())
print(error.sum())

print(a.mean())
print(a.var())
