import torch
import torch.nn as nn
import torch.nn.functional as F
class SAE(nn.Module):
    def __init__(self,d_model,d_sae):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_sae)
        # bias为负，大部分神经元处于关闭状态
        nn.init.constant_(self.linear1.bias, -0.1)

        self.linear2 = nn.Linear(d_sae,d_model)

    def encoder(self,x):
        x = F.relu(self.linear1(x))
        return x

    def decoder(self,x):
        x = self.linear2(x)
        return x

    def forward(self,x):
        h = self.encoder(x)
        result = self.decoder(h)
        return result,h

#加Decoder权重归一化防止坍缩
#激活率低于 1% 的神经元超过 80%，说明已经开始坍缩
#with torch.no_grad():
    #sae.linear2.weight.data = sae.linear2.weight.data / sae.linear2.weight.data.norm(dim=1, keepdim=True)

#监控激活率把 h 变成 0/1 掩码

#重构损失（MSE测输出与输入差距）和稀疏损失（对h取L1正则使h值尽可能小）
#总损失 = 重构损失 + 系数 * 稀疏损失