import numpy as np
from generator import Generator,GreedySampler
from model import DecoderOnly

from config import *

model = DecoderOnly(d_model, n_heads, d_ff, num_blocks, vocab_size)

model.load_weight(save_path)

sampler = GreedySampler()
generator = Generator(sampler)

start_ids = np.array([[1]])

generate_ids = generator.generate(model,start_ids,2,max_len)
print(generate_ids)