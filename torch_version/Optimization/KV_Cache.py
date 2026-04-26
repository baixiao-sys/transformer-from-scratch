import torch

def quant_fp8(x):
    max_abs = x.abs().max(dim=-1, keepdim=True)[0]
    scale = max_abs / 448.0
    x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return x_fp8, scale

def dequant_fp8(x_fp8, scale):
    return x_fp8.to(torch.bfloat16) * scale