import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    def __init__(self,dk,max_seq_len=2048,base=10000,num_fope=0,cutoff=0.1):
        super().__init__()
        self.dk = dk
        self.max_seq_len = max_seq_len
        self.base = base

        theta = torch.arange(0,dk,2).float()/self.dk
        freq = self.base**(-theta)

        if num_fope != 0:
            freqs = []
            for k in range(num_fope):
                scale = (k+1)/num_fope
                theta = scale*(torch.arange(0,dk,2).float()/self.dk)
                freq = self.base**theta
                freqs.append(freq)
            freqs = torch.stack(freqs).mean(dim=0)
            #低频过滤
            freqs = torch.where(freqs < cutoff, torch.zeros_like(freqs), freqs)
            self.register_buffer("inv_freq",freqs)

        self.register_buffer("inv_freq",freq)

    def forward(self,x):
        B, n_heads, seq_len, d_k = x.shape
        device = x.device

        #float精度可改
        pos = torch.arange(seq_len,dtype=torch.float32,device=device)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((freqs,freqs),dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos,sin

    def rotate_half(self,x):
        x1, x2 = x.chunk(2, dim=-1)#拦腰斩断
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self,q, k, cos, sin):
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        return q_rot, k_rot


class SPE(nn.Module):
    def __init__(self,d_model,max_len=2048,dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,max_seq_len):
        pos_emb = self.pe[:, :max_seq_len]
        return pos_emb
        #增加dropout
        #return self.dropout(pos_emb)



class EmbPos(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        #可学习位置编码
        #self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
        # SPE，目前已关闭
        self.pos_encoding = SPE(embed_dim, max_seq_len)

    def forward(self,token_ids):
        token_emb = self.embedding(token_ids)
        seq_len = token_ids.shape[1]
        # SPE
        #token_emb = token_emb + self.pos_emb[:,:token_ids.size(1)]
        #pos_emb = self.pos_encoding.forward(seq_len)
        """
        #return token_emb + pos_emb
        """
        return token_emb

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head,max_seq_len=2048,num_fope=0):
        super().__init__()
        assert  d_model%n_head==0
        self.d_model = d_model
        self.n_head = n_head
        self.dk = d_model // n_head

        self.Wq = nn.Linear(d_model,d_model)
        self.Wk = nn.Linear(d_model,d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model,d_model)

        """
        torch实现RoPE
        """
        self.rope = RoPE(self.dk,max_seq_len,num_fope=num_fope)


    def forward(self,q,k,v,mask,return_attn = False,kv_cache=None):
        batch_size,seq_len = q.shape[:2]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = q.view(batch_size, seq_len, self.n_head, self.dk).transpose(1,2)
        k = k.view(batch_size, seq_len, self.n_head, self.dk).transpose(1,2)
        v = v.view(batch_size, seq_len, self.n_head, self.dk).transpose(1,2)

        """
        旋转位置编码
        """
        cos,sin = self.rope(q)
        q,k = self.rope.apply_rope(q,k,cos,sin)

        #kv缓存
        if kv_cache is not None:
            prev_k, prev_v = kv_cache

            k = torch.cat([prev_k, k], dim=2) #seq_len
            v = torch.cat([prev_v, v], dim=2)

        new_kv_cache = (k, v)

        """
        手动注意力方便返回权重生成热力图
        """
        if return_attn:
            attn_score = q @ k.transpose(-1, -2) / (k.size(-1) ** 0.5)
            if mask is not None:
                #_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
                #attn_score = attn_score.masked_fill(~_mask, float("-inf"))
                attn_score = attn_score+mask
            attn_weight = F.softmax(attn_score, dim=-1)
            attn_output = attn_weight @ v
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.Wo(attn_output)
            return output, attn_weight,new_kv_cache

        attn = F.scaled_dot_product_attention(q, k, v,attn_mask=None,is_causal=True)

        attn_output = attn.transpose(1,2).contiguous()
        attn_output = attn_output.view(batch_size,-1,self.d_model)
        output = self.Wo(attn_output)
        """
        ----------------------------
        用cache优化缓存
        ---------------------------
        """
        return output,new_kv_cache

class FFN(nn.Module):
    def __init__(self,d_model,d_ff,dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input):
        input = F.relu(self.linear1(input))
        #return self.linear2(input)
        #加入dropout
        return self.linear2(self.dropout(input))


class Transformer_Block_PreLN(nn.Module):
    def __init__(self,d_model,n_head,d_ff,dropout=0.1,num_fope=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = MultiHeadAttention(d_model,n_head,num_fope=num_fope)

        self.ffn = FFN(d_model, d_ff)

        self.dropout = nn.Dropout(dropout)

    def forward(self,input,mask=None,return_attn=False,kv_cache=None):
        if return_attn:
            attn_out,attn_weight,kv_cache = self.attn(self.norm1(input), self.norm1(input), self.norm1(input), mask,return_attn,kv_cache)
            input = self.dropout(attn_out) + input

            self.ffn_out = self.ffn(self.norm2(input))

            input = input + self.dropout(self.ffn_out)
            return input,attn_weight,kv_cache

        attn_out,kv_cache = self.attn(self.norm1(input), self.norm1(input), self.norm1(input), mask,kv_cache)
        input = input + self.dropout(attn_out)

        self.ffn_out = self.ffn(self.norm2(input))

        input = input + self.dropout(self.ffn_out)
        return input,kv_cache