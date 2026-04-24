import torch
import torch.nn.functional as F
from config import return_attn
from DataVisualization.attn_map import plot_attention_map


def generate_text(model, prompt,tokenizer, device,temperature=0.8,top_k=50,top_p = 0.9, max_new_tokens=100):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    tokens = list(prompt)
    with torch.no_grad():
        for step in range(max_new_tokens):
            if return_attn:
                logits,attn_maps,_ = model(input_ids,return_attn=return_attn)
                """
                热力图
                """
                #if step == 90:
                    #plot_attention_map(attn_maps, tokens, 0, 2)
            else:
                logits,_ = model(input_ids)
            logits = logits[:, -1, :]/temperature#温度缩放

            #topk采样
            if top_k>0:
                v, _ = torch.topk(logits,top_k,dim=-1)
                logits[logits<v[:,[-1]]] = -float("inf")

            #topp采样
            if 1>top_p>0:
                sorted_logits,sorted_indices = torch.sort(logits,descending=True)
                cumsum_prob = torch.cumsum(torch.softmax(sorted_logits,dim=-1),dim=-1)
                sorted_indices_to_remove = cumsum_prob>top_p
                sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
                sorted_indices_to_remove[...,0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:,indices_to_remove] = -float("inf")


            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)#按概率抽，抽1个下个字
            input_ids = torch.cat((input_ids, next_token), dim=1)

            #新token加入tokens列表
            tokens.append(tokenizer.decode(next_token.item()))


    output_tokens = input_ids[0].tolist()
    output_text = tokenizer.decode(output_tokens)
    return output_text

def generate_text_with_kv(model, prompt,tokenizer, device,temperature=0.8,top_k=50,top_p = 0.9, max_new_tokens=100):
    kv_caches = None
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    tokens = list(prompt)
    with torch.no_grad():
        for step in range(max_new_tokens):
            if return_attn:
                logits,attn_maps,kv_caches = model(input_ids[:,-1:],return_attn=return_attn,kv_caches=kv_caches)
                """
                热力图
                """
                #if step == 90:
                    #plot_attention_map(attn_maps, tokens, 3, 0)
            else:
                logits,kv_caches = model(input_ids[:,-1:],kv_caches=kv_caches)
            logits = logits[:, -1, :]/temperature#温度缩放

            #topk采样
            if top_k>0:
                v, _ = torch.topk(logits,top_k,dim=-1)
                logits[logits<v[:,[-1]]] = -float("inf")

            #topp采样
            if 1>top_p>0:
                sorted_logits,sorted_indices = torch.sort(logits,descending=True)
                cumsum_prob = torch.cumsum(torch.softmax(sorted_logits,dim=-1),dim=-1)
                sorted_indices_to_remove = cumsum_prob>top_p
                sorted_indices_to_remove[...,1:] = sorted_indices_to_remove[...,:-1].clone()
                sorted_indices_to_remove[...,0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:,indices_to_remove] = -float("inf")


            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)#按概率抽，抽1个下个字
            input_ids = torch.cat((input_ids, next_token), dim=1)

            #新token加入tokens列表
            tokens.append(tokenizer.decode(next_token.item()))


    output_tokens = input_ids[0].tolist()
    output_text = tokenizer.decode(output_tokens)
    return output_text