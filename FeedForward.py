import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionwiseFeedForward(nn.Module):
    """两层全连接层，中间使用ReLU激活函数"""

    def __init__(self, d_model, hidden, dropout=0.1):
        """d_model: 输入和输出的维度"""
        """hidden: 隐藏层的维度"""
        """dropout: dropout的概率"""
        super().__init()
        self.fc1 = nn.Linear(d_model, hidden)  #第一层线性变换
        self.fc2 = nn.Linear(hidden, d_model)  #第二层线性变换
        self.dropout = nn.Dropout(dropout)    #dropout层

    def forward(self,x):
        """x: 形状为(batch_size, seq_len, d_model)"""
        x = self.fc1(x)
        x = F.relu(x)  #ReLU激活函数
        x = self.dropout(x) #dropout
        x = self.fc2(x)
        return x  #返回输出张量
