import torch
import torch.nn as nn
import torch.nn.functional as F
import math # For mathematical operations

class LayerNorm(nn.Module):
    def __init__(self, x,eps=1e-6):
        """x: 形状为(batch_size, seq_len, d_model)"""
        """eps: 防止除零的小常数(稳定性)"""
        super().__init__()
        self.eps = eps  #数值稳定性常数
        self.gama = nn.Parameter(torch.ones(x.size(-1)))  #缩放参数，初始化为1，形状为(d_model,)
        self.beta = nn.Parameter(torch.zeros(x.size(-1)))  #平移参数，初始化为0，形状为(d_model,)
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  #计算均值，形状为(batch_size, seq_len, 1)
        std = x.std(-1, unbiased=False,keepdim=True)    #计算标准差，形状为(batch_size, seq_len, 1)
        norm_x = (x - mean) / (std + self.eps)  #标准化
        output = self.gama * norm_x + self.beta  #缩放和平移
        return output  #返回归一化后的输出


###<***注意***>
#LayerNorm 是对最后一个维度进行归一化处理的，即对每个时间步的特征维度进行归一化
#归一化后，会对输出进行output = self.gama * norm_x + self.beta 缩放和平移
#gama和beta是可学习的参数，允许模型调整归一化后的分布：
##模型既能享受归一化带来的数值稳定性，又不会失去灵活性。