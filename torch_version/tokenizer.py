import torch
import json

from torch.utils.data import IterableDataset

from torch_version.config import file_path,save_vocab_path,BPE_save_path,BPE_load_path

import os

def stream_load_text_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        #大数据分批次边读边生成词表
        for line in f:
            #line = line.strip()
            if line:
                yield line

def load_text_data(file_path,tokenizer,max_len=128):
    all_ids = []
    for line in stream_load_text_data(file_path):
        ids = tokenizer.encode(line)
        all_ids.extend(ids)

    input_ids = []
    target_ids = []

    for i in range(0, len(all_ids) - max_len, max_len):
        input_seq = all_ids[i: i + max_len]
        target_seq = all_ids[i + 1: i + max_len + 1]

        input_ids.append(input_seq)
        target_ids.append(target_seq)

    input_ids = torch.LongTensor(input_ids)
    target_ids = torch.LongTensor(target_ids)

    return input_ids,target_ids

class CharTokenizer:
    def __init__(self,file_path=file_path):
        chars = []
        for line in stream_load_text_data(file_path=file_path):
            for char in line:
                if char:
                    chars.append(char)
        self.vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(chars)
        self.vocab_size = len(self.vocab)
        self.stoi = {c:i for i, c in enumerate(self.vocab)}
        self.itos = {i:c for i, c in enumerate(self.vocab)}
        self.save_vocab(save_vocab_path=save_vocab_path)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        if isinstance(ids, int):
            return self.itos[ids]
        return ''.join([self.itos[i] for i in ids])

    def get_vocab_size(self):
        return self.vocab_size

    def save_vocab(self, save_vocab_path="char_vocab.json"):
        with open(save_vocab_path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": {int(k): v for k, v in self.itos.items()},
                "vocab_size": self.vocab_size
            }, f, ensure_ascii=False, indent=2)#保持中文输出和2个缩进

    def load_vocab(self, load_vocab_path="char_vocab.json"):
        with open(load_vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = data["vocab_size"]


class BPE:
    def __init__(self,vocab_size=13000,min_frequency=2,special_tokens=["<pad>", "<unk>", "<s>", "</s>"]):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.tokenizer = None

    def train_save(self,file_path,BPE_save_path=BPE_save_path):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE as BPEModel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        self.tokenizer = Tokenizer(BPEModel(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(vocab_size=self.vocab_size,min_frequency=self.min_frequency,special_tokens=self.special_tokens)

        self.tokenizer.train([file_path], trainer)

        self.tokenizer.save(BPE_save_path)
        print(f"小词表训练完成保存为{BPE_save_path}")

    def load(self,BPE_load_path=BPE_load_path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(BPE_load_path)
        return self.tokenizer

    def encode(self,text):
        assert self.tokenizer is not None
        return self.tokenizer.encode(text).ids

    def decode(self,ids):
        assert self.tokenizer is not None
        if isinstance(ids,(int,float)):
            ids = [int(ids)]
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        assert self.tokenizer is not None
        return self.tokenizer.get_vocab_size()


class StreamDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, seq_len=256,total_steps_cache="total_steps_cache.json"):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.total_steps_cache = total_steps_cache

    def __iter__(self):
        buffer = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer.encode(line)
                buffer.extend(tokens)

                while len(buffer) >= self.seq_len + 1:
                    input_ids = buffer[:self.seq_len]
                    target_ids = buffer[1:self.seq_len + 1]
                    input_ids = torch.tensor(input_ids)
                    target_ids = torch.tensor(target_ids)
                    yield input_ids,target_ids
                    buffer = buffer[self.seq_len:]


    def get_len(self,batch_size=8,epochs=10):
        buffer = []
        total = 0

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = self.tokenizer.encode(line)
                buffer.extend(tokens)
                while len(buffer) >= self.seq_len + 1:
                    total += 1
                    print(total)
                    buffer = buffer[self.seq_len:]

        steps_per_epoch = total // batch_size
        total_steps = steps_per_epoch*epochs
        if not os.path.exists(self.total_steps_cache):
            print("第一次运行，预统计总样本数...")
            with open(self.total_steps_cache, "w") as w:
                json.dump({"total_samples": total_steps}, w)
        else:
            print("加载预统计的总样本数...")
            with open(self.total_steps_cache, "r") as r:
                total_steps = json.load(r)["total_samples"]
        return total_steps