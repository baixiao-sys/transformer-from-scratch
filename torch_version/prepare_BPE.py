from tokenizer import BPE

"""
tokenizer = BPE()
tokenizer.train_save("../assets/text/TinyStories-train.txt")
print("BPE 词表训练完成，已保存为 BPE_text.json")
"""
from tokenizer import StreamDataset, BPE


tokenizer = BPE()
tokenizer.load("BPE_text.json")


dataset = StreamDataset("../assets/text/TinyStories-train.txt", tokenizer, seq_len=256)
dataset.get_len(batch_size=8, epochs=10)
print("预统计完成！现在可以直接运行 train.py 了")
#1766296手动存一下