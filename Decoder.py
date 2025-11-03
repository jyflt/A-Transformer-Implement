import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For mathematical operations
import MultiHead_Attention, FeedForward, LayerNorm, Embedding, Encoder

class DecoderLayer(nn.Module):
    def __init__(self,d_model, n_head, hidden, dropout=0.1):
        super().__init__()
        self.attention1 = MultiHead_Attention.MultiHeadAttention(d_model, n_head)  #自注意力
        self.attention2 = MultiHead_Attention.MultiHeadAttention(d_model, n_head)  #编码器-解码器注意力
        self.ffn = FeedForward.PositionwiseFeedForward(d_model, hidden,dropout=dropout)
        self.norm1 = LayerNorm.LayerNorm(torch.zeros(1,1,d_model))  #初始化LayerNorm
        self.norm2 = LayerNorm.LayerNorm(torch.zeros(1,1,d_model))
        self.norm3 = LayerNorm.LayerNorm(torch.zeros(1,1,d_model))
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, enc_out, src_mask,tgt_mask ,tgt_attn_mask):
        attn1, _ = self.attention1(x,x,x,mask=tgt_mask,attn_mask=tgt_attn_mask)  #自注意力机制
        attn1 = self.dropout(attn1)  #应用dropout
        x = self.norm1(x + attn1)  #残差连接和归一化
        attn2,_ = self.attention2(x,enc_out,enc_out,mask=src_mask)  #编码器-解码器注意力机制
        attn2 = self.dropout(attn2)  #应用dropout
        x = self.norm2(x + attn2)  #残差连接和归一化
        ffn_out = self.ffn(x)  #前馈神经网络
        ffn_out = self.dropout(ffn_out)  #应用dropout
        out = self.norm3(x + ffn_out)  #残差连接和归一化
        return out
    
class Decoder(nn.Module):
    def __init__ (self,vocab_size, max_len, d_model, n_head, hidden, num_layers,device, dropout=0.1):
        super().__init__()
        self.embedding = Embedding.TransformerEmbedding(vocab_size, d_model, max_len, dropout, device)
        #堆叠多个解码器层
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, hidden, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, vocab_size)  #输出层，将d_model映射到词汇表大小
        
    def forward(self, x, enc_out,src_mask, tgt_mask,tgt_attn_mask,dropout):
        #嵌入
        dec = self.embedding(x)  #获取嵌入表示
        #通过每个解码器层
        for layer in self.layers:
            dec = layer(dec, enc_out, src_mask, tgt_mask,tgt_attn_mask,dropout)  #逐层传递
        out = self.fc(dec)    #输出层
        return out
