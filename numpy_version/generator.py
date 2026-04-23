import numpy as np

"""
自回归生成
"""
class Sampler:
    def sample(self,logits):
        pass

class GreedySampler(Sampler):
    def sample(self,logits):
        next_token_id = np.argmax(logits,axis=-1)
        return next_token_id

class Generator:
    def __init__(self,sampler):
        self.sampler = sampler

    def generate(self,model,start_ids,eos_token_id=None,max_len=100):
        model.eval()
        cur_ids = start_ids
        for _ in range(max_len):
            seq_len = cur_ids.shape[1]
            causal_mask = np.tril(np.ones((seq_len, seq_len)))
            logits = model.forward(cur_ids,causal_mask)
            last_logits = logits[:,-1,:]
            next_token_id = self.sampler.sample(last_logits)
            next_token_id = np.expand_dims(next_token_id,1)
            cur_ids = np.concatenate([cur_ids,next_token_id],1)
            if eos_token_id is not None and (next_token_id==eos_token_id).all():
                    break
        return cur_ids