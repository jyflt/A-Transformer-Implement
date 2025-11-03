import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For mathematical operations
import MultiHead_Attention
import FeedForward
import LayerNorm, Embedding
class EncoderLayer(nn.Module):
    def __init__(self,d_model, n_head, hidden, dropout=0.1):
        super().__init__()
        self.attention = MultiHead_Attention.MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForward.PositionwiseFeedForward(d_model, hidden,dropout=dropout)
        self.norm1 = LayerNorm.LayerNorm(torch.zeros(1,1,d_model))  #初始化LayerNorm
        self.norm2 = LayerNorm.LayerNorm(torch.zeros(1,1,d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        _emb = x
        attn,_ = self.attention(_emb, _emb, _emb, mask=mask)  #自注意力机制
        _attn = self.dropout(attn)  #应用dropout
        out1 = self.norm1(_emb + _attn)  #残差连接和归一化
        ffn_out = self.ffn(out1)  #前馈神经网络
        _ffn_out = self.dropout(ffn_out)  #应用dropout
        out2 = self.norm2(out1 + _ffn_out)  #残差连接和归一化
        return out2
    
class Encoder(nn.Module):
    def __init__(self,vocab_size, max_len, d_model, n_head, hidden, num_layers,device, dropout=0.1):
        super().__init__()
        self.embedding = Embedding.TransformerEmbedding(vocab_size, d_model, max_len, dropout, device)
        #堆叠多个编码器层
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, hidden, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm.LayerNorm(torch.zeros(1,1,d_model))  #最终归一化层

    def forward(self, x, mask):
        #嵌入
        emb = self.embedding(x)  #获取嵌入表示
        out = emb
        #通过每个编码器层
        for layer in self.layers:
            out = layer(out, mask)  #逐层传递
        
        out = self.norm(out)  #最终归一化
        return out