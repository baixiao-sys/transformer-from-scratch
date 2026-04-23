import numpy as np
from model import DecoderOnly
from modules import cross_entropy, AdamW
from config import *

Model = DecoderOnly
min_loss = float("inf")

model = Model(d_model, n_heads, d_ff, num_blocks, vocab_size)

params = model.get_params()
print(f"模型总参数数量: {len(params)}")
for i, p in enumerate(params):
    print(f"参数 {i + 1}: 形状 {p.data.shape}, 类型 {type(p)}")

input_ids = np.array([
    [1, 2, 3, 4, 5, 6, 7, 8],
    [1, 2, 3, 4, 5, 6, 7, 8]
], dtype=np.int64)
target_ids = np.array([
    [2, 3, 4, 5, 6, 7, 8, 9],
    [2, 3, 4, 5, 6, 7, 8, 9]
], dtype=np.int64)
"""
input_ids,target_ids = load_data(file_path,max_len)
"""

causal_mask = np.tril(np.ones((seq_len, seq_len)))
optimizer = AdamW(params)

for epoch in range(epochs):
    model.train()
    logits = model.forward(input_ids, causal_mask)
    print(f"\nLogits 形状: {logits.shape}")

    loss, d_logits = cross_entropy(logits, target_ids)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")

    model.backward(input_ids, d_logits)

    """
    #监测使用
    embed_grad_norm = np.linalg.norm(model.emb_pos.embedding.embedding_matrix.grad)
    print(f"Embedding梯度范数: {embed_grad_norm:.6f}")

    linear_grad_norm = np.linalg.norm(model.final_linear.weight.grad)
    print(f"Final Linear梯度范数: {linear_grad_norm:.6f}")
    """

    optimizer.step()
    print("参数更新完成")
    optimizer.zero_grad()
    print("梯度已清零")

    if loss<min_loss:
        model.save_weight(save_path)
        min_loss = loss
        print(f"发现更优模型：loss = {min_loss:.4f}，已保存→best_model.npz")