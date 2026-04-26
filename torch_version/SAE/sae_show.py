import json
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torch_version.model import DecoderOnly
from torch_version.config import d_model, n_heads, seq_len, d_ff, num_blocks, num_fope, batch_size, lr, return_attn
from torch_version.tokenizer import StreamDataset,BPE

from torch_version.SAE.SAE import SAE

from torch_version.DataVisualization.attn_map import plot_attention_map

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print("当前设备:", device)



tokenizer = BPE()
#Tinystories
tokenizer.load(BPE_load_path="../prepare/BPE_text.json")
print("BPE已加载")

prompt = "Once upon a time."
input_ids = tokenizer.encode(prompt)
input_ids = torch.tensor([input_ids]).to(device)

vocab_size = tokenizer.get_vocab_size()
model = DecoderOnly(vocab_size=vocab_size,d_model=d_model, n_heads=n_heads,max_seq_len=seq_len, d_ff=d_ff, num_blocks=num_blocks,num_fope=num_fope)
model = model.to(device)
model.load_weight("../weights/best_model.pth")
model.eval()
#冻结参数
for params in model.parameters():
    params.requires_grad = False
#SAE
d_sae=1024#参数设置多少合适
sae = SAE(d_model=d_model,d_sae=d_sae).to(device)
sae.load_state_dict(torch.load("sae.pth"))
sae.eval()

return_attn = True
kv_caches = None
max_new_tokens = 100
with torch.no_grad():
    tokens = list(prompt)
    if return_attn:
        logits, attn_maps, kv_caches = model(input_ids[:, -1:], return_attn=return_attn, kv_caches=kv_caches)
        """
        热力图
        """
        plot_attention_map(attn_maps, tokens, 3, 0)
    else:
        logits, kv_caches = model(input_ids[:, -1:], kv_caches=kv_caches)

    ffn_out = model.blocks[0].ffn_out
    sae_recon, sae_h = sae(ffn_out)

    activation = sae_h[0].mean(0).cpu().numpy()
    plt.bar(range(d_sae), activation)
    plt.title("SAE Activation")
    plt.show()