import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For mathematical operations

#将输入转为语义空间
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """vocab_size: 词汇表的大小"""
        """d_model: 词嵌入的维度"""
        super().__init__(vocab_size, d_model,padding_idx=1)  #调用父类的初始化方法,padding_idx=1表示索引1对应的词是填充词


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """d_model: 词嵌入的维度"""
        """max_len: 句子的最大长度"""
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model).to(device)  #初始化一个全零的张量，形状为(max_len, d_model)
        self.encoding.requires_grad = False  #<**关键**> 不需要对位置编码进行梯度更新
        pos = torch.arange(0, max_len,device = device)
        pos = pos.float().unsqueeze(1)  #将位置索引转换为浮点数，并调整形状为(max_len, 1)
        _2i = torch.arange(0, d_model, step=2,device = device).float()  #生成一个从0到d_model的偶数索引张量
        self.encoding[:,0::2] = torch.sin(pos / (10000**(_2i / d_model)))  #对偶数位置应用正弦函数
        self.encoding[:,1::2] = torch.cos(pos / (10000**(_2i / d_model)))  #对奇数位置应用余弦函数

    def forward(self, x):
        """x: 输入的句子批嵌入张量，形状为(batch_size, seq_len, d_model)"""
        """返回位置编码张量，形状为(batch_size, seq_len, d_model)"""
        batch_size, seq_len = x.size(0), x.size(1)
        #返回位置编码, 形状为(batch_size, seq_len, d_model)
        return self.encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
    
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_out, device):
        """vocab_size: 词汇表的大小"""
        """d_model: 词嵌入的维度"""
        """max_len: 句子的最大长度"""
        """drop_out: dropout的概率"""
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model,padding_idx=1)  #初始化词嵌入层
        self.position_embedding = PositionalEmbedding(d_model, max_len, device)  #初始化位置嵌入层
        self.dropout = nn.Dropout(p=drop_out)  #初始化dropout层
    
    def forward(self, x):
        """x: 输入的句子批张量，形状为(batch_size, seq_len)"""
        """返回嵌入张量，形状为(batch_size, seq_len, d_model)"""
        token_emb = self.token_embedding(x)  #获取词嵌入，形状为(batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(token_emb)  #获取位置嵌入，形状为(batch_size, seq_len, d_model)
        emb = token_emb + pos_emb  #将词嵌入和位置嵌入相加
        emb = self.dropout(emb)  #应用dropout
        return emb  #返回最终的嵌入张量

###<***注意***>
#1、句子的对齐在预处理阶段完成，这里不处理对齐问题
#2、填充词的索引设为1，在TokenEmbedding中通过padding_idx参数指定


#阶段	                                操作	                                                 说明
#数据准备	                 将不同长度的句子补齐（pad）到相同长度	                          得到统一形状的输入张量
#TokenEmbedding	      将每个tokenid映射为向量；padding_idx 对应的向量恒为0	                 对齐后可安全输入模型
#注意力mask	                在self-attention阶段遮掉 <PAD> 部分	                     模型不会学习无意义的填充部分