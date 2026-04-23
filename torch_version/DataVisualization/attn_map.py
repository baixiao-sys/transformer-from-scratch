import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_attention_map(attn_weights, tokens, layer_idx=0, head_idx=0, save_path=None):
    """
    attn_weights: [B, n_heads, T, T]
    tokens: 字符串列表，长度 T
    """
    # 取 batch=0，指定层和头的注意力
    #print("attn_weights shape:", attn_weights.shape)
    attn = attn_weights[layer_idx][0, head_idx].cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(attn, cmap='viridis', vmin=0, vmax=1)

    # 加上 token 标签
    plt.xticks(np.arange(len(tokens)), tokens, rotation=90, fontsize=8)
    plt.yticks(np.arange(len(tokens)), tokens, fontsize=8)

    plt.title(f"Attention Map | Layer {layer_idx} | Head {head_idx}")
    plt.colorbar()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()