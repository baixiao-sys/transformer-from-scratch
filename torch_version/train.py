import math
import time
from config import *

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from model import DecoderOnly
from torch_version.tokenizer import CharTokenizer, BPE, StreamDataset

from tokenizer import load_text_data
from torch_version.DataVisualization.view_test import loss_curve

import os
import json

print("当前GPU:", torch.cuda.get_device_name(0))
print("CUDA可用:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

scaler = torch.amp.GradScaler('cuda',enabled=(dtype == 'float16'))

min_loss = float("inf")

"""
字符级分词器
"""
#tokenizer = CharTokenizer()
tokenizer = BPE()
#tokenizer.train_save("../assets/text/TinyStories-train.txt")
tokenizer.load("torch_version/prepare/BPE_text.json")
print("BPE已加载")

"""
#字符级
#input_ids,target_ids = load_text_data(file_path,tokenizer,max_len)
input_ids,target_ids = load_text_data("../assets/text/TinyStories-train.txt",tokenizer,max_len)
input_ids = input_ids.to(device)
target_ids = target_ids.to(device)
print("数据所在设备:", target_ids.device)
#pad_id = tokenizer.encode("<pad>")
"""

vocab_size = tokenizer.get_vocab_size()


model = DecoderOnly(vocab_size=vocab_size,d_model=d_model, n_heads=n_heads,max_seq_len=seq_len, d_ff=d_ff, num_blocks=num_blocks,num_fope=num_fope)
model = model.to(device)
model.train()



#批次迭代
#dataset = TensorDataset(input_ids, target_ids)
dataset = StreamDataset("../assets/text/TinyStories-train.txt",tokenizer,seq_len)
                                                                   #流式就得设置为0        #放GPU上
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)


train_losses = []  #用于画图/分析

optimizer = torch.optim.AdamW(model.parameters(),lr)

"""
断点续训
"""
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

#选择续训模型
#model_name = "rope"
#model_name = "fope"
#model_name = "tiny_stories_rope"
model_name = "tiny_stories_fope"

start_epoch = 0
start_micro_step = 0
checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_last.pt")

if os.path.exists(checkpoint_path):
    print(f"加载断点：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    train_losses = checkpoint["train_losses"]
    start_micro_step = checkpoint.get("micro_step", 0)
    min_loss = checkpoint.get("min_loss", float("inf"))
    print(f"从epoch:{start_epoch}|micro_step:{start_micro_step}继续训练|最优Loss:{min_loss:.4f}")

"""
学习率预热
"""
with open("prepare/total_steps_cache.json", "r") as f:
    total_steps = json.load(f)["total_samples"]
    print("total_step已加载")

#total_steps = dataset.get_len(batch_size=batch_size,epochs=epochs)
#total_steps = 100000 * epochs
warm_steps = int(total_steps*0.05)
def lr_scheduler(step):
    if step <warm_steps:
        return (step+1)/(warm_steps+1)
    decay_radio = (step-warm_steps)/(total_steps-warm_steps)
    assert 0<=decay_radio<=1
    coeff = 0.5*(1.0+math.cos(decay_radio*math.pi))
    return coeff
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler)

start_time = time.time()

for epoch in range(start_epoch,epochs):
    #print(f"进入 Epoch {epoch + 1}，当前 batch 总数：{len(dataloader)}")
    print(f"进入 Epoch {epoch + 1}")
    epoch_start_time = time.time()
    batch_count = 0
    optimizer.zero_grad()

    total_loss = 0.0
    period_total_loss = 0.0
    micro_start_time = time.time()
    for micro_step,(input_ids,target_ids) in  enumerate(dataloader):

        if micro_step<start_micro_step:
            continue

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        with torch.autocast(device_type="cuda", dtype=dtype,enabled=True):
            logits,_ = model(input_ids)
            logits = logits.view(-1,logits.shape[-1])
            loss = F.cross_entropy(logits,target_ids.view(-1),ignore_index=-1)
            loss = loss / gradient_accumulation_steps


        scaler.scale(loss).backward()
        total_loss += loss.item()*gradient_accumulation_steps

        batch_count += 1
        if (micro_step + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer)  # 裁剪前需反缩放梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        period_total_loss += loss.item() * gradient_accumulation_steps


        if micro_step % log_interval == 0:
            current_loss = loss.item()
            #print(f"[Epoch {epoch + 1}]|Batch{micro_step}|Loss:{current_loss:.4f}")
            train_losses.append(current_loss)

        if micro_step%save_micro_step==0 and micro_step > 0 and micro_step!=start_micro_step:
            current_avg_loss = period_total_loss / save_micro_step

            #model.save_weight(save_weight_path)
            if current_avg_loss<min_loss:
                min_loss = current_avg_loss
                model.save_weight("best_model.pth")
                print(f"更优模型loss={min_loss:.4f}")
            else:
                print(f"当前Loss:{current_avg_loss:.4f} | 最优Loss:{min_loss:.4f}")
            period_total_loss = 0.0

            torch.save({
                "epoch": start_epoch,
                "micro_step": micro_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "min_loss": min_loss
            }, checkpoint_path)

            print(f"微步{micro_step}已保存模型|当前Loss:{current_avg_loss:.4f}|最优Loss:{min_loss:.4f}")
            print(f"当前{micro_step}耗时{time.time()-micro_start_time:.2f}s")
            micro_start_time = time.time()





    avg_loss = total_loss / batch_count
    elapsed = time.time() - epoch_start_time
    print(f"\nEpoch {epoch + 1}/{epochs}|平均Loss:{avg_loss:.4f}|耗时:{elapsed:.2f}s\n")

    if avg_loss<min_loss:
        model.save_weight(save_weight_path)
        min_loss = avg_loss
        print(f"更优模型loss={min_loss:.4f}")

    if epoch%check_step == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "train_losses":train_losses,
        }, os.path.join(checkpoint_dir, f"{model_name}_checkpoint_last.pt"))
        print("epoch保存了一下")
        pass


time = time.time() - start_time
print(f"\n总耗时:{time:.2f}s")

loss_curve(train_losses)
#loss_curve(train_losses,loss_curve_save_path)