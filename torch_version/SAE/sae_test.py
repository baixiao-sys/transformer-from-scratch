import json
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_version.model import DecoderOnly
from torch_version.config import d_model,n_heads,seq_len,d_ff,num_blocks,num_fope,batch_size,lr
from torch_version.tokenizer import StreamDataset,BPE

from torch_version.SAE.SAE import SAE


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print("当前设备:", device)



tokenizer = BPE()
#Tinystories
tokenizer.load(BPE_load_path="../prepare/BPE_text.json")
print("BPE已加载")

vocab_size = tokenizer.get_vocab_size()
model = DecoderOnly(vocab_size=vocab_size,d_model=d_model, n_heads=n_heads,max_seq_len=seq_len, d_ff=d_ff, num_blocks=num_blocks,num_fope=num_fope)
model = model.to(device)
model.load_weight("../weights/best_model.pth")
model.eval()
#冻结参数
for params in model.parameters():
    params.requires_grad = False
#SAE
d_sae=1024#参数设置多少合适
sae = SAE(d_model=d_model,d_sae=d_sae).to(device)
#激活计数器
counter = torch.zeros(d_sae,device=device)
total_tokens = 0

dataset = StreamDataset("../../assets/text/TinyStories-train.txt",tokenizer,seq_len)
                                                                   #流式就得设置为0        #放GPU上
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)


optimizer = torch.optim.AdamW(sae.parameters(),lr)

with open("../prepare/total_steps_cache.json", "r") as f:
    total_steps = json.load(f)["total_samples"]
    print("total_step已加载")
warm_steps = int(total_steps*0.05)
def lr_scheduler(step):
    if step <warm_steps:
        return (step+1)/(warm_steps+1)
    decay_radio = (step-warm_steps)/(total_steps-warm_steps)
    assert 0<=decay_radio<=1
    coeff = 0.5*(1.0+math.cos(decay_radio*math.pi))
    return coeff
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lr_scheduler)


min_loss = float("inf")
epochs = 1

train_losses = []

#加载保存点
start_epoch = 0
start_micro_step = 0
model_name = "sae_test"
checkpoint_dir = "weights"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_checkpoint_last.pt")
if os.path.exists(checkpoint_path):
    print(f"加载断点：{checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint["epoch"]
    start_step = checkpoint.get("step", 0)
    sae.load_state_dict(checkpoint["sae_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses = checkpoint["train_losses"]
    min_loss = checkpoint.get("min_loss", float("inf"))
    print(f"从epoch:{start_epoch}|step:{start_step}继续训练|最优Loss:{min_loss:.4f}")


for epoch in range(start_epoch,epochs):
    print(f"进入 Epoch {epoch}")

    for step,(input_ids,target_ids) in enumerate(dataloader):
        if step<start_step:
            continue
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        with torch.no_grad():
            model(input_ids)


        ffn_out = []
        for block in model.blocks:
            ffn_out.append(block.ffn_out)


        x = ffn_out[0].to(device)

        x_recon,h = sae(x)

        h_active = (h > 0).float()
        counter += h_active.sum(dim=(0, 1))
        total_tokens += h.shape[0] * h.shape[1]

        loss_recon = F.mse_loss(x_recon,x)
        loss_spar = h.abs().mean()
        #loss_spar要调小防止坍塌
        #太小了下降又难
        loss = loss_recon+0.005*loss_spar

        """
        h_slice = h[0, 0, :20]
        print("神经元亮灯情况 (前20):")
        print((h_slice > 0).int().cpu().numpy())
        """

        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 1000 == 0:
            train_losses.append(loss.item())
            if loss<min_loss:
                torch.save(sae.state_dict(), "sae.pth")
                min_loss = loss
                print(f"更优模型loss={min_loss:.4f}")
            print(f"Step {step}/{total_steps}, Recon Loss: {loss_recon.item():.4f}, Sparse Loss: {loss_spar.item():.4f},Loss:{loss.item():.4f}")

            active_rate = counter / total_tokens
            dead = (active_rate < 0.001).sum().item()
            print(f"死神经元: {dead}/{d_sae}")
            print(f"激活率: {active_rate.mean().item():.4f}")

            torch.save({
                "epoch": epoch,
                "step": step,
                "sae_state_dict": sae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_losses,
                "min_loss": min_loss
            }, checkpoint_path)

        optimizer.zero_grad()

#35000步时开Sparse Loss开始下降
#60000步时Sparse Loss有点稳定在0.46左右不下降
#0.002在10400步又不下降稳在0.28