import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def kv_cache_picture(speed):
    labels = ["关闭KV Cache", "开启KV Cache"]

    plt.figure(figsize=(7, 5))
    # 画直方图
    bars = plt.bar(
        labels, speed,
        color=["#ff6b6b", "#4ecdc4"],  # 红/青 对比色
        width=0.6, edgecolor="black"
    )
    plt.bar_label(bars, fmt="%d tokens/s", fontsize=12, fontweight="bold")

    plt.title("KV Cache开启/关闭推理速度对比", fontsize=14, fontweight="bold")
    plt.ylabel("推理速度(tokens/s)", fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()