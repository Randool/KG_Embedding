import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from config import SEP, Dir, encode
from data_loader import load_batch_triple
from model.config import *
from model.TransE import TransE


def marginLoss(pos, neg, margin):
    return torch.sum(F.relu(pos - neg + margin))


def normLoss(embeds, dim=1):
    norm = torch.sum(embeds ** 2, dim=dim, keepdim=True)
    return torch.sum(F.relu(norm - FloatTensor([1.0])))


def train(model: nn.Module, train_file, val_file, lr, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma=0.1)
    # loss_function = MarginLoss()
    # if CUDA:
    #     loss_function.cuda()
    _margin = FloatTensor([margin])
    train_losses, val_losses, total_time = [], [], 0

    def Stage(data, stage: str) -> float:
        """ train/val 复用函数 """
        (ph, pr, pt), (nh, nr, nt) = data
        model.zero_grad()
        pos, neg = model(ph, pr, pt, nh, nr, nt)
        # loss = loss_func() + embeddings
        loss = marginLoss(pos, neg, _margin)
        ent_embed = model.ent_embed(torch.cat([ph, pt, nh, nt]))
        rel_embed = model.rel_embed(torch.cat([pr, nr]))
        loss += normLoss(ent_embed) + normLoss(rel_embed)
        if stage == "train":
            loss.backward()
            optimizer.step()
        return loss.item()

    
    log_file = open("train.log", "w", encoding=encode)
    text = "====== Start ======"
    print(text)
    log_file.write("{}\n".format(text))

    for epoch in range(epochs):
        tic = time.time()
        scheduler.step()
        
        # prepare data loader
        train_loader = load_batch_triple(train_file, batch_size)
        val_loader = load_batch_triple(val_file, batch_size)
        
        # Train state
        train_loss, val_loss = 0, 0
        train_cnt, val_cnt = 0, 0   # 用于计算loss平均值
        for data in train_loader:
            train_cnt += data[0][0].shape[0]    # 累计batch
            train_loss += Stage(data, "train")
        # Test state
        with torch.no_grad():
            for data in val_loader:
                val_cnt += data[0][0].shape[0]    # 累计batch
                val_loss += Stage(data, "val")
        
        # Statistic
        ave_train_loss = train_loss / train_cnt
        ave_val_loss = val_loss / val_cnt
        train_losses.append(ave_train_loss)
        val_losses.append(ave_val_loss)
        time_frag = (time.time() - tic) / 60
        total_time += time_frag
        
        # Visual
        info_train = "Epoch {:#3d} | Train loss: {:#.4f}".format(epoch, ave_train_loss)
        info_val = "Validation loss: {:#.4f}".format(ave_val_loss)
        info = "{} | {} | Cost: {:#.2f} min".format(info_train, info_val, time_frag)
        print(info)
        log_file.write("{}\n".format(info))
        tic = time.time()
    
    text = "====== Finish ======\t{} min".format(total_time)
    print(text)
    log_file.write("{}\n".format(text))

    # Save model
    torch.save(model, "embed.pt")
    # Draw
    import matplotlib.pyplot as plt
    x = [i for i in range(len(train_losses))]
    plt.plot(x, train_losses)
    plt.plot(x, val_losses)
    plt.legend(["train", "val"])
    plt.savefig("loss.png")


if __name__ == "__main__":
    # entity_cnt, relation_cnt, embed_dim
    with open("note.log") as f:
        ent_cnt = int(f.readline().split(SEP)[1])
        rel_cnt = int(f.readline().split(SEP)[1])
        print("Entity count {}\nRelation count {}".format(ent_cnt, rel_cnt))

    model = TransE(ent_cnt, rel_cnt, embed_dim)
    if CUDA:
        model.cuda()
    
    train_file = os.path.join(Dir, "train_triples_index.txt")
    val_file = os.path.join(Dir, "val_triples_index.txt")

    train(model, train_file, val_file, lr, epochs)
