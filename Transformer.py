import torch, torch.nn as nn
import Encoder
import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx, #原语言填充标记索引
                 tgt_pad_idx, #目标语言填充标记索引
                 src_vocab_size,
                 tgt_vocab_size,
                 max_len,
                 d_model,
                 n_head,
                 hidden,
                 num_layers,
                 device,
                 dropout=0.1):
        super().__init__()
        self.encoder = Encoder.Encoder(src_vocab_size, max_len, d_model, n_head, hidden, num_layers, device, dropout)
        self.decoder = Decoder.Decoder(tgt_vocab_size, max_len, d_model, n_head, hidden, num_layers, device, dropout)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
    def make_pad_mask(self, q,k,pad_idx_q, pad_idx_k):
        """q: 查询序列张量，形状为(batch_size, len_q)"""
        """k: 键序列张量，形状为(batch_size, len_k)"""
        """QK^T返回一个形状为(batch_size, n_head, len_q, len_k)的掩码张量"""
        len_q , len_k = q.size(1), k.size(1)
        #ne:not equal(返回一个布尔张量，表示不等于的位置)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)  #（batch_size,1,len_q,1）便于传播
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)  #（batch_size,1,1,len_k）便于传播
        q = q.repeat(1,1,1,len_k)  #（batch_size,1,len_q,len_k）
        k = k.repeat(1,1,len_q,1)  #（batch_size,1,len_q,len_k）
        mask = q & k  #“与”运算，形状为(batch_size,1,len_q,len_k)
        return mask  #返回填充掩码张量
    
    def make_causal_mask(self,q,k):
        """针对目标序列的因果掩码"""
        mask = torch.tril(torch.ones((q.size(1),k.size(1)))).type(torch.BoolTensor).to(self.device)  #下三角矩阵
        return mask  #返回因果掩码张量
    
    def forward(self, src, tgt,dropout):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx).to(self.device)  #源语言填充掩码
        tgt_pad_mask = self.make_pad_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx).to(self.device)  #目标语言填充掩码
        tgt_causal_mask = self.make_causal_mask(tgt, tgt).to(self.device)
        enc_out = self.encoder(src, src_mask)  #编码器输出
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_pad_mask, tgt_causal_mask,dropout)  #解码器输出
        return dec_out  #返回最终输出
    