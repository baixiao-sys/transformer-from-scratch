import time
from config import d_model,n_heads,d_ff,num_blocks,seq_len,temperature,top_k,top_p
from model import DecoderOnly
import torch

from tokenizer import CharTokenizer,BPE

from generator import generate_text
from torch_version.generator import generate_text_with_kv

tokenizer = BPE()
tokenizer.load("BPE_text.json")
vocab_size =tokenizer.vocab_size

model = DecoderOnly(vocab_size,d_model, n_heads, d_ff, num_blocks,seq_len)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.load_weight("best_model.pth")
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
print(f"{time.time()-starttime:.2f}s")
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
    max_new_tokens=200
)
print(f"{time.time()-starttime:.2f}s")
"""