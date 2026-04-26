import time
from config import d_model,n_heads,d_ff,num_blocks,seq_len,temperature,top_k,top_p
from model import DecoderOnly
import torch

from tokenizer import CharTokenizer,BPE

from generator import generate_text
from torch_version.generator import generate_text_with_kv

tokenizer = BPE()
tokenizer.load("prepare/BPE_text.json")
vocab_size =tokenizer.vocab_size

model = DecoderOnly(vocab_size,d_model, n_heads, d_ff, num_blocks,seq_len)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_weight("weights/best_model.pth")
model.eval()

starttime=time.time()
text = generate_text(
    model=model,
    prompt="once upon a time",
    tokenizer=tokenizer,
    device=device,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    max_new_tokens=100
)
time1 = time.time()-starttime
print(f"{time1:.2f}s")
print(text)

"""
starttime=time.time()
text = generate_text_with_kv(
    model=model,
    prompt="a",
    tokenizer=tokenizer,
    device=device,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    max_new_tokens=100
)
time2 = time.time()-starttime
print(f"{time2:.2f}s")

"""
#可视化部分
"""
speed1 = 100/time1
speed2 = 100/time2
speed = [speed1,speed2]
from DataVisualization.kv_cache import kv_cache_picture
kv_cache_picture(speed)
"""