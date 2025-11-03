import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model, n_head,dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.n_d = self.d_model // self.n_head    #每个头的维度
        self.w_q = nn.Linear(self.d_model,self.d_model,bias=False)
        self.w_k = nn.Linear(self.d_model, self.d_model,bias=False)
        self.w_v = nn.Linear(self.d_model, self.d_model,bias=False)
        self.w_combine = nn.Linear(self.d_model, self.d_model,bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask,attn_mask=None):
        batch,time, dimension = q.shape
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q = q.view(batch, time, self.n_head, self.n_d).transpose(1, 2)  #调整形状为(batch, n_head, time, n_d)
        k = k.view(batch, time, self.n_head, self.n_d).transpose(1, 2)
        v = v.view(batch, time, self.n_head, self.n_d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.n_d)  #计算注意力分数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 自定义 attn_mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:          # (Lq, Lk) -> (1,1,Lq,Lk)
                am = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:        # (B, Lq, Lk) -> (B,1,Lq,Lk)
                am = attn_mask.unsqueeze(1)
            else:
                raise ValueError("attn_mask 维度必须是 2 或 3")
            if am.dtype == torch.bool:
                scores = scores.masked_fill(~am, float("-inf"))  #把这些位置填上负无穷
            else:
                scores = scores + am  # 浮点掩码（如上三角=-inf）

        attn = self.softmax(scores)  #计算注意力权重
        attn = self.attn_dropout(attn)  #应用dropout
        output = torch.matmul(attn, v)  #加权求和得到输出
        output = output.transpose(1,2).contiguous().view(batch, time, self.d_model)  #调整形状回(batch, time, d_model)
        output = self.w_combine(output)  #线性变换得到最终输出
        return output, attn  #返回输出和注意力权重
    
