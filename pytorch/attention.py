import torch.nn as nn
import math
import torch
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads)
       
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

    def forward(self,key,query,value,mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
       
        k = self.key_matrix(key)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        k_adjusted = k.transpose(-1,-2)
        product = torch.matmul(q, k_adjusted)
      
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        product = product / math.sqrt(self.single_head_dim)

        scores = F.softmax(product, dim=-1)
 
        scores = torch.matmul(scores, v)

        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)
        
        output = self.out(concat)
       
        return output
