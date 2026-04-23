vocab_size = 50257
epochs = 10
d_model = 16
n_heads = 2
d_ff = 64
num_blocks = 4#2
batch_size = 8
seq_len = 8
max_len = 128

gradient_accumulation_steps = 4

lr = 1e-4
temperature = 0.8
top_k=50
top_p = 0.9

save_path = "numpy_model.npz"
load_path = "numpy_model.npz"



