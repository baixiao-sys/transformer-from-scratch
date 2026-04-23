import matplotlib.pyplot as plt

def draw_fope_rope(rope_ppl,fope_ppl,seq_lens=[128,256,512,1024,2048,4096]):
    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, rope_ppl, marker='o', label='RoPE', color='#ff7f0e', linewidth=2)
    plt.plot(seq_lens, fope_ppl, marker='s', label='FoPE', color='#1f77b4', linewidth=2)

    # 美化和标注
    plt.title("PPL vs Context Length (RoPE vs FoPE)", fontsize=14, pad=15)
    plt.xlabel("Context Length", fontsize=12)
    plt.ylabel("Perplexity (PPL)", fontsize=12)
    plt.xticks(seq_lens)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    #plt.savefig("rope_vs_fope_ppl.png", dpi=150, bbox_inches='tight')
    plt.show()