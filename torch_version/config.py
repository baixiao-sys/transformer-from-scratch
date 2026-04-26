vocab_size = 50257
epochs = 2
d_model = 256
n_heads = 8
d_ff = 1024
num_blocks = 6#2
batch_size = 16
seq_len = 256
max_len = 256

gradient_accumulation_steps = 8

lr = 1e-4
temperature = 0.8
top_k=50
top_p = 0.9
#load_path = "../numpy_version/model.npz"
file_path = "1.txt"
#save_path = "best_model.pth"
save_weight_path = "weights/TinyStories_model.pth"
load_weight_path = "weights/TinyStories_model.pth"

return_attn = True

"""
from tokenizers import Tokenizer
# 加载
tokenizer = Tokenizer.from_file("../prepare/BPE_text.json")
# 编码
ids = tokenizer.encode("all_txt.txt").ids
# 解码
text = tokenizer.decode(ids)
vocab_size = tokenizer.get_vocab_size()
"""

log_interval = 10  #每10个batch打印一次
loss_curve_save_path = "loss_curve.png"

BPE_save_path = "prepare/BPE_text.json"
BPE_load_path = "prepare/BPE_text.json"

save_vocab_path = "prepare/char_vocab.json"
load_vocab_path = "prepare/char_vocab.json"

num_fope = 0#为0时自动关掉FoPE

rope_model_weight_path = "weights/rope_model_weight.pth"
fope_model_weight_path = "weights/fope_model_weight.pth"

check_step = 10

save_micro_step = 2000

