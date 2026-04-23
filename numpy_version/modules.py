import numpy as np

class RoPE:
    def __init__(self,d_model,base=10000):
        self.base = base
        self.theta = 1/(self.base ** (np.arange(0,d_model,2) / d_model))

    def forward(self,q):
        seq_len = q.shape[1]
        pos = np.arange(0,seq_len)
        pos = np.expand_dims(pos,1)
        theta = np.expand_dims(self.theta,0)
        freq = pos*theta
        cos,sin = np.cos(freq),np.sin(freq)
        cos = np.expand_dims(cos, 0)
        sin = np.expand_dims(sin, 0)
        q_rot = np.zeros_like(q)
        q_rot[...,0::2] = q[...,0::2]*cos-q[...,1::2]*sin
        q_rot[...,1::2] = q[...,0::2]*sin+q[...,1::2]*cos

        self.cos = cos
        self.sin = sin
        return q_rot

    def backward(self,dout):
        d_q = np.zeros_like(dout)
        d_q[...,0::2] = dout[...,0::2]*self.cos-dout[...,1::2]*self.sin
        d_q[...,1::2] = dout[...,1::2]*self.sin+dout[...,1::2]*self.cos
        return d_q


class SPE:
    def __init__(self,embed_dim,seq_len):
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        #预计算，只在初始化时计算一次
        self.pos_emb = self._precompute_pos_emb()

    def _precompute_pos_emb(self):
        #频率项
        i = np.arange(0,self.embed_dim, 2)
        div = np.exp(np.log(10000) * (-2) * i / self.embed_dim)
        #算pos
        pos = np.arange(0, self.seq_len).reshape(-1, 1)
        pos_emb = np.zeros((self.seq_len, self.embed_dim))
        pos_emb[:, 0::2] = np.sin(div * pos)
        pos_emb[:, 1::2] = np.cos(div * pos)
        return pos_emb  #[seq_len, embed_size]

    def forward(self,seq_len):
        return np.expand_dims(self.pos_emb[:seq_len], axis=0)#[:seq_len]，动态长度

class Embedding:
    def __init__(self,vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        #Xavier Norm初始化
        self.embedding_matrix = Parameter(np.random.randn(vocab_size, embed_dim) * np.sqrt(2 / (vocab_size + embed_dim)))

    def forward(self,token_ids):
        return self.embedding_matrix.data[token_ids]

    def backward(self,token_ids,d):
        B,S,D = d.shape
        d_embed = np.zeros((self.vocab_size,self.embed_dim))
        token_ids_flat = token_ids.reshape(-1)
        d_flat = d.reshape(-1, self.embed_dim)
        np.add.at(d_embed,token_ids_flat,d_flat)
        d_embed /= B*S
        self.embedding_matrix.grad += d_embed
        return d_embed

class EmbPos:
    def __init__(self, vocab_size, embed_dim, max_seq_len=512):
        self.embedding = Embedding(vocab_size,embed_dim)
        #SPE
        self.pos_encoding = SPE(embed_dim, max_seq_len)

    def forward(self,token_ids):
        batch_size, seq_len = token_ids.shape
        token_emb = self.embedding.forward(token_ids)
        """
        pos_emb = self.pos_encoding.forward(seq_len)
        #使用了RoPE替代SPE
        #return token_emb + pos_emb
        """
        return token_emb

    def backward(self,token_ids,dout):
        return self.embedding.backward(token_ids,dout)

    def get_params(self):
        return [self.embedding.embedding_matrix]

class Parameter:
    data:np.ndarray
    grad:np.ndarray

    def __init__(self,data):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad.fill(0)


class Linear:
    def __init__(self,in_dims:int,out_dims:int,bias=False):
        self.in_dims = in_dims
        self.out_dims = out_dims
        #He初始化W->ReLu
        scale = np.sqrt(6/in_dims)
        W = np.random.uniform(-scale,scale,(in_dims,out_dims))

        self.weight = Parameter(W)
        self.bias = None
        if bias:
            self.bias =Parameter(np.zeros((self.out_dims,)))

    def forward(self,input):
        self.input = input
        out = input@self.weight.data
        if self.bias is not None:
            out += self.bias.data
        return out

    def backward(self,d_out):
        self.weight.grad += np.sum(self.input.transpose(0,2,1)@d_out,axis=0)
        if self.bias is not None:
            self.bias.grad += np.sum(d_out,(0,1))
        return d_out@self.weight.data.T

    def __call__(self, input):
        return self.forward(input)

def softmax(input):
    # 保持每个数都是负数，每个e_x都小于1，不会数据爆炸溢出
    input = input-np.max(input,axis=-1,keepdims=True)
    e_x = np.exp(input)
    sum_e_x = np.sum(e_x,axis=-1,keepdims=True)
    sum_e_x = np.maximum(sum_e_x,1e-12)
    return e_x/sum_e_x

class MultiHeadAttention:
    def __init__(self,d_model,n_head):
        assert  d_model%n_head==0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.Wq = Linear(d_model,d_model)
        self.Wk = Linear(d_model,d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model,d_model)

        self.rope = RoPE(d_model)

    def Attention(self,q, k, v, mask=None):

        d_k = q.shape[-1]
        attn_score = np.matmul(q, k.swapaxes(2, 3)) / np.sqrt(d_k)

        # 因果掩码-----------------------------------
        if mask is not None:
            if mask.ndim == 2:
                mask = np.expand_dims(np.expand_dims(mask, 0), 0)
            attn_score = np.where(mask == 0, -1e9, attn_score)
        # -----------------------------------------

        attn_weight = softmax(attn_score)
        out = np.matmul(attn_weight, v)
        return out, attn_weight,attn_score

    def forward(self,input,mask):
        # 在cross_attention时分开，来源不同
        self.X = input
        batch_size,seq_len = input.shape[:2]
        q = self.Wq(input)
        k = self.Wk(input)
        v = self.Wv(input)

        """
        旋转位置编码
        """
        q = self.rope.forward(q)
        k = self.rope.forward(k)

        q = q.reshape(batch_size, seq_len, self.n_head, self.d_model // self.n_head)
        k = k.reshape(batch_size, seq_len, self.n_head, self.d_model // self.n_head)
        v = v.reshape(batch_size, seq_len, self.n_head, self.d_model // self.n_head)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        self.Q = q
        self.K = k
        self.V = v

        out,attn_weight,attn_score = self.Attention(q,k,v,mask)

        outconcat = out.transpose(0,2,1,3).reshape(batch_size,-1,self.d_model)

        output = self.Wo(outconcat)

        self.attn_weight = attn_weight
        self.attn_score = attn_score
        self.outconcat = outconcat

        return output

    def backward(self,d_out):
        B, S, D = d_out.shape

        d_outconcat = self.Wo.backward(d_out)

        d_outconcat = d_outconcat.reshape(B, S, self.n_head, self.d_k).transpose(0, 2, 1, 3)

        d_attn_weight = d_outconcat@self.V.transpose(0, 1, 3, 2)
        d_V = self.attn_weight.transpose(0, 1, 3, 2)@d_outconcat
        #交叉注意力q和k来源不一样，Sq!=Sk，所以attn_weight需要转置

        d_attn_score = self.attn_weight * (d_attn_weight -  np.sum(self.attn_weight * d_attn_weight, axis=-1, keepdims=True))

        d_Q = d_attn_score@self.K/np.sqrt(D)
        d_K = d_attn_score.transpose(0,1,3,2)@self.Q/np.sqrt(D)

        #合并为了backward对齐维度
        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(B, S, D)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(B, S, D)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(B, S, D)

        d_Q = self.rope.backward(d_Q)
        d_K = self.rope.backward(d_K)

        d_xfromQ = self.Wq.backward(d_Q)
        d_xfromK = self.Wk.backward(d_K)
        d_xfromV = self.Wv.backward(d_V)
        d_X = d_xfromQ+d_xfromK+d_xfromV

        return d_X

    def __call__(self,input,mask):
        return self.forward(input,mask)
"""
GeLU在CPU有点慢了，numpy先用着ReLU
"""
class GeLU:
    def __init__(self):
        #self.c = np.float32(0.044714998453855516)
        self.c = 0.044715
        self.sq = np.sqrt(2 / np.pi)

    def forward(self,input):
        self.input = input
        return 0.5*input*(1+np.tanh(self.sq*(input+self.c*input**3)))

    def backward(self,dout):
        x = self.input
        tanh_psi = np.tanh(self.sq*(x+self.c*x**3))
        dx = 0.5*(1+tanh_psi)+0.5*x*(1-tanh_psi**2)*self.sq*(1+self.c*3*x**2)
        return dx*dout

class ReLU:
    def forward(self,input):
        self.input = input
        out = np.maximum(0,input)
        return out

    def backward(self,d_out):
        d_input = d_out * (self.input > 0)
        return d_input

    def __call__(self,input):
        return self.forward(input)

class FFN:
    def __init__(self,d_model,d_ff,bias=True):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.active_func = ReLU()#可替换GeLU

    def forward(self,input):
        func_out = self.active_func(self.linear1(input))
        result2 = self.linear2(func_out)
        self.relu_out = func_out
        return result2

    def backward(self,d_out):
        d_relu = self.linear2.backward(d_out)
        d_linear1_out = self.active_func.backward(d_relu)
        dx = self.linear1.backward(d_linear1_out)
        return dx

    def __call__(self, input):
        return self.forward(input)

class LayerNorm:
    def __init__(self,d_model,eps=1e-6):
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))
        self.eps = eps

    def forward(self,input):
        mean = np.mean(input,-1,keepdims=True)
        var = np.mean((input-mean)**2,-1,keepdims=True)
        x_hat = (input-mean)/np.sqrt(var+self.eps)
        out = self.gamma.data*x_hat + self.beta.data

        self.cache = (input, mean, var, x_hat)
        return out #[batch_size,seq_len,d_model]

    def backward(self,d_out):
        input,mean,var,x_hat = self.cache
        d_beta = np.sum(d_out,(0,1))
        d_gamma = np.sum(d_out*x_hat,(0,1))
        #参数更新部分
        self.beta.grad += d_beta
        self.gamma.grad += d_gamma

        d_model = input.shape[-1]
        sigma = np.sqrt(var+self.eps)
        d_xhat = d_out*self.gamma.data
        sum_d_xhat = np.sum(d_xhat,-1,keepdims=True)

        sum_d_xhat_x = np.sum(d_xhat*x_hat,-1,keepdims=True)
        d_x = ((d_model*d_xhat)-sum_d_xhat-(x_hat*sum_d_xhat_x))/(d_model*sigma)#LN反向传播dx公式
        return d_x

    def get_params(self):
        return [self.gamma,self.beta]

    def __call__(self, input):
        return self.forward(input)

class Dropout:
    def __init__(self,drop_prob=0.1):
        self.drop_prob = drop_prob
        self.mask = None
        self.train = True

    def set_train(self):
        self.train = True

    def set_eval(self):
        self.train = False

    def forward(self,input):
        if not self.train:
            return input

        mask = np.random.rand(*input.shape)>self.drop_prob
        out = input*mask/(1-self.drop_prob)#保持期望值不变

        self.mask = mask
        return out

    def backward(self,d_out):
        if not self.train:
            return d_out
        dx = d_out*self.mask/(1-self.drop_prob)
        return dx

    def __call__(self, input):
        return self.forward(input)


class Transformer_Block_PreLN:
    def __init__(self,d_model,n_head,d_ff,drop_prob=0.1):
        self.d_model = d_model
        self.n_head = n_head

        self.norm1 = LayerNorm(self.d_model)
        self.norm2 = LayerNorm(self.d_model)

        self.attn = MultiHeadAttention(self.d_model,self.n_head)

        self.ffn = FFN(d_model, d_ff)

        self.dropout1 = Dropout(drop_prob)
        self.dropout2 = Dropout(drop_prob)

    def forward(self,input,mask=None):
        # 注意力
        norm_1 = self.norm1(input)
        attn = self.attn(norm_1,mask)

        attn_drop = self.dropout1(attn)
        input = input+attn_drop

        norm_2 = self.norm2(input)
        ffn = self.ffn(norm_2)
        ffn_drop = self.dropout2(ffn)
        input = input+ffn_drop
        return input

    def backward(self,d_out):
        d_dropout2 = self.dropout2.backward(d_out)
        d_ffn = self.ffn.backward(d_dropout2)
        d_norm2 = self.norm2.backward(d_ffn)
        d_res1 = d_out + d_norm2
        d_dropout1 = self.dropout1.backward(d_res1)
        d_attn = self.attn.backward(d_dropout1)
        d_norm1 = self.norm1.backward(d_attn)
        d_x = d_res1 + d_norm1
        return d_x

    def get_params(self):
        params = []
        params.extend(self.norm1.get_params())
        params.extend(self.norm2.get_params())
        params.append(self.attn.Wq.weight)
        params.append(self.attn.Wk.weight)
        params.append(self.attn.Wv.weight)
        params.append(self.attn.Wo.weight)
        if self.attn.Wq.bias is not None:
            params.append(self.attn.Wq.bias)
        if self.attn.Wk.bias is not None:
            params.append(self.attn.Wk.bias)
        if self.attn.Wv.bias is not None:
            params.append(self.attn.Wv.bias)
        if self.attn.Wo.bias is not None:
            params.append(self.attn.Wo.bias)
        params.append(self.ffn.linear1.weight)
        params.append(self.ffn.linear2.weight)
        if self.ffn.linear1.bias is not None:
            params.append(self.ffn.linear1.bias)
        if self.ffn.linear2.bias is not None:
            params.append(self.ffn.linear2.bias)
        return params

    def train(self):
        self.dropout1.set_train()
        self.dropout2.set_train()

    def eval(self):
        self.dropout1.set_eval()
        self.dropout2.set_eval()

    def __call__(self,input,mask=None):
        return self.forward(input,mask)

class Optimizer:
    def __init__(self,params,lr=1e-3):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self,params,lr=1e-3):
        super().__init__(params,lr)

    def step(self):
        for p in self.params:
            p.data -= p.grad*self.lr

class AdamW(Optimizer):
    def __init__(self,params,lr=1e-4,beta1=0.9,beta2=0.99,weight_decay=0.01,eps=1e-8):
        super().__init__(params,lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = {p:np.zeros_like(p.data)for p in params}
        self.v = {p: np.zeros_like(p.data) for p in params}
        self.weight_decay = weight_decay
        self.eps = eps

    def step(self):
        self.t += 1

        for p in self.params:
            if p.grad is None:
                continue
            self.m[p] = self.beta1*self.m[p]+(1-self.beta1)*p.grad
            self.v[p] = self.beta2*self.v[p]+(1 - self.beta2) * (p.grad**2)
            m_hat = self.m[p]/(1-self.beta1**self.t)
            v_hat = self.v[p]/(1-self.beta2**self.t)
            update = m_hat/np.sqrt(v_hat+self.eps)
            if self.weight_decay!=0:
                update = update+self.weight_decay*p.data
            p.data -= self.lr*update

def cross_entropy(logits,target,smoothing=0.1):
    B,S,D = logits.shape
    p = softmax(logits)
    target_onehot = np.zeros_like(p)

    # 标签平滑
    target_onehot[np.arange(B)[:, None], np.arange(S)[None, :], target] = 1-smoothing
    target_onehot += smoothing/D

    loss = -np.log(p+1e-12)*target_onehot
    loss =np.sum(loss,-1)
    loss = np.mean(loss)

    d_logits = p-target_onehot
    d_logits = d_logits / (B * S)  #匹配均值梯度

    return loss,d_logits
