import matplotlib.pyplot as plt
import torch
import os
from torch_version.model import DecoderOnly
from torch_version.config import *
import numpy as np

def plot_weight_distribution(model, max_layers=10):
    """
    绘制模型权重分布
    """
    plt.figure(figsize=(14, 8))
    layer_count = 0

    for name, param in model.named_parameters():
        # 只看权重，不看偏置
        if "weight" in name and param.requires_grad:
            # 只画前N层，避免图太多
            if layer_count >= max_layers:
                break

            # 把权重转成numpy
            weights = param.data.cpu().numpy().flatten()

            hist, bins = np.histogram(weights, bins=100, density=True)
            bin_centers = (bins[1:] + bins[:-1]) / 2
            plt.plot(bin_centers, hist, linewidth=1.5, label=name[:45])

            # 画直方图
            #plt.hist(weights, bins=100, alpha=0.5, label=name[:50])

            layer_count += 1

    plt.title("Model Weight Distribution (Xavier初始化)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.show()

def plot_single_layer_weight(model, layer_name="linear"):
    for name, param in model.named_parameters():
        if layer_name in name and "weight" in name:
            weights = param.data.cpu().numpy().flatten()
            plt.hist(weights, bins=100)
            plt.title(name)
            plt.show()
            return

"""
model = DecoderOnly(vocab_size,d_model, n_heads, d_ff, num_blocks)
model.load_weight("best_model.pth")
plot_weight_distribution(model)
#plot_single_layer_weight(model, "q")
"""


#os.makedirs("weight_plots", exist_ok=True)

def plot_and_save_weights(model, epoch=0, max_layers=10):
    """
    自动绘制并保存权重分布图
    保存路径：weight_plots/epoch_0.png
    """
    plt.switch_backend('Agg')

    plt.figure(figsize=(16, 8))

    layer_count = 0
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            if layer_count >= max_layers:
                break

            # 展平权重
            w = param.data.cpu().numpy().flatten()
            # 绘制直方图
            plt.hist(w, bins=80, alpha=0.5, label=name[:45])

            layer_count += 1

    plt.title(f"Weight Distribution | Epoch {epoch}")
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)

    plt.savefig(f"weight_plots/epoch_{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close()

def loss_curve(train_losses,save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
    plt.show()
