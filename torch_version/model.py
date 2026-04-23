import torch
import torch.nn as nn

from numpy_version.config import seq_len
from torch_version.modules import EmbPos,Transformer_Block_PreLN

class Model(nn.Module):
    def __init__(self,vocab_size,d_model, n_heads, d_ff, num_blocks ,max_seq_len,dropout=0.1,num_fope=0):
        super().__init__()

        self.emb_pos = EmbPos(vocab_size,d_model,max_seq_len)

        self.blocks = nn.ModuleList([Transformer_Block_PreLN(d_model, n_heads, d_ff,dropout,num_fope=num_fope) for _ in range(num_blocks)])
        self.final_linear = nn.Linear(d_model, vocab_size)
        #Xavier初始化
        self.apply(self._init_weights_Xavier)

        #生成因果掩码
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.float32))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float("-inf"))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self,token_ids,mask=None,return_attn=False,kv_caches=None):
        """
        正弦位置编码,已经在类中关掉(pos_emb+token_emb)，只返回token_emb
        """
        input = self.emb_pos.forward(token_ids)
        seq_len = input.shape[1]

        mask = self.causal_mask[:,:,:seq_len,:seq_len]

        if kv_caches is None:
            new_kv_caches = None
            attn_maps = [] if return_attn else None
            for block in self.blocks:
                if return_attn:
                    input, attn_weight,_ = block.forward(input, mask=mask, return_attn=return_attn, kv_cache=None)
                    attn_maps.append(attn_weight)
                else:
                    input,_ = block.forward(input, mask=mask, return_attn=return_attn, kv_cache=None)
            logits = self.final_linear(input)

            if return_attn:
                return logits, attn_maps,new_kv_caches
            else:
                return logits,new_kv_caches

        else:
            new_kv_caches = []
            attn_maps = [] if return_attn else None
            for block in self.blocks:
                if return_attn:
                    input, attn_weight,kv_cache = block.forward(input, mask, return_attn, kv_caches)
                    attn_maps.append(attn_weight)
                else:
                    input,kv_cache = block.forward(input, mask, return_attn, kv_caches)
                    new_kv_caches.append(kv_cache)
            logits = self.final_linear(input)
            if return_attn:
                return logits, attn_maps,new_kv_caches
            else:
                return logits,new_kv_caches

    def save_weight(self, path="model.npz"):
        torch.save(self.state_dict(), path)

    def load_weight(self, path="model.npz"):
        state_dict = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))#指定设备
        self.load_state_dict(state_dict)

    def _init_weights(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights_Xavier(self,module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
        elif isinstance(module,nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

class Encoder:
    def my_func(self):
        raise NotImplementedError("Encoder 暂未实现")
    pass

class Decoder:
    def my_func(self):
        raise NotImplementedError("Decoder 暂未实现")
    pass

class DecoderOnly(Model):
    def __init__(self, vocab_size ,d_model, n_heads, d_ff, num_blocks,max_seq_len,dropout=0.1,num_fope=0):
        super().__init__(vocab_size ,d_model, n_heads, d_ff, num_blocks,max_seq_len,dropout,num_fope)
