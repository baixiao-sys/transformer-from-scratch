import numpy as np
from modules import EmbPos, Transformer_Block_PreLN, Linear

class Model:
    def __init__(self,d_model, n_heads, d_ff, num_blocks, vocab_size ,max_seq_len=512):

        self.emb_pos = EmbPos(vocab_size,d_model,max_seq_len)

        self.blocks = [Transformer_Block_PreLN(d_model, n_heads, d_ff) for _ in range(num_blocks)]
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self,token_ids,mask=None):
        input = self.emb_pos.forward(token_ids)

        for block in self.blocks:
            input = block.forward(input,mask)

        logits = self.final_linear(input)
        return logits

    def backward(self,token_ids,d_logits):
        d_out = self.final_linear.backward(d_logits)

        for block in reversed(self.blocks):
            d_out = block.backward(d_out)

        d_emb = self.emb_pos.backward(token_ids,d_out)
        return d_emb

    def get_params(self):
        params = []
        params.append(self.emb_pos.embedding.embedding_matrix)

        for block in reversed(self.blocks):
            block_params= block.get_params()
            params.extend(block_params)

        params.append(self.final_linear.weight)
        if self.final_linear.bias is not None:
            params.append(self.final_linear.bias)
        return params

    def train(self):
        for block in self.blocks:
            block.train()

    def eval(self):
        for block in self.blocks:
            block.eval()

    def save_weight(self, path="numpy_model.npz"):

        weights = {}

        def collect(obj, prefix=""):
            for name, val in obj.__dict__.items():
                if isinstance(val, np.ndarray):
                    weights[prefix + name] = val
                elif hasattr(val, "__dict__"):
                    collect(val, prefix + name + ".")
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        collect(item, prefix + name + f"[{i}].")
        collect(self)

        np.savez(path, **weights)

    def load_weight(self, path="numpy_model.npz"):

        data = np.load(path)

        def load(obj, prefix=""):
            for name, val in obj.__dict__.items():
                key = prefix + name
                if key in data:
                    obj.__dict__[name] = data[key]
                elif hasattr(val, "__dict__"):
                    load(val, key + ".")
                elif isinstance(val, list):
                    for i, item in enumerate(val):
                        load(item, prefix + name + f"[{i}].")

        load(self)

    def __call__(self, token_ids,mask=None):
        return self.forward(token_ids,mask)

class Encoder:
    def my_func(self):
        raise NotImplementedError("Encoder 暂未实现")
    pass

class Decoder:
    def my_func(self):
        raise NotImplementedError("Decoder 暂未实现")
    pass

class DecoderOnly(Model):
    def __init__(self,d_model, n_heads, d_ff, num_blocks, vocab_size ,max_seq_len=512):
        super().__init__(d_model, n_heads, d_ff, num_blocks, vocab_size ,max_seq_len)
