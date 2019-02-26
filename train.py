import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from model.config import *
from model.TransE import TransE

from config import SEP
from data_loader import load_batch_triple


class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, pos, neg, margin):
        return torch.sum(self.relu(pos - neg + margin))


def normLoss(embeds, dim=1):
    norm = torch.sum(embeds ** 2, dim=dim, keepdim=True)
    return torch.sum(F.relu(norm - FloatTensor([1.0])))


def train(model: nn.Module, train_file, val_file, lr, epochs):
    print("====== Start ======")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = MarginLoss()
    if CUDA:
        loss_function.cuda()
    _margin = FloatTensor([margin])

    def stage(data, grad: bool):
        # for train and validate
        (ph, pr, pt), (nh, nr, nt) = data
        model.zero_grad()
        pos, neg = model(ph, pr, pt, nh, nr, nt)
        # loss = loss_func() + embeddings
        loss = loss_function(pos, neg, _margin)
        ent_embed = model.ent_embed(torch.cat([ph, pt, nh, nt]))
        rel_embed = model.rel_embed(torch.cat([pr, nr]))
        loss += normLoss(ent_embed) + normLoss(rel_embed)
        if grad:
            loss.backward()
            optimizer.step()
        return loss.item()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        tic = time.time()
        
        # prepare data loader
        train_loader = load_batch_triple(train_file, batch_size)
        val_loader = load_batch_triple(val_file, batch_size)
        
        # Train state
        train_loss, val_loss = 0, 0
        train_cnt, val_cnt = 0, 0   # 用于计算loss平均值
        for data in train_loader:
            train_cnt += data[0][0].shape[0]    # 累计batch
            train_loss += stage(data, True)
        # Test state
        with torch.no_grad():
            for data in val_loader:
                val_cnt += data[0][0].shape[0]    # 累计batch
                val_loss += stage(data, False)
        
        # Visual
        print("Epoch {:#3d} | Train loss: {:#.4f}".format(epoch, (train_loss / train_cnt)), end=" | ")
        print("Validation loss: {:#.4f}", end=" | ".format((val_loss / val_cnt)))
        print("Cost: {:#.2f} min".format((time.time() - tic) / 60))
        train_losses.append(train_loss / train_cnt)
        val_losses.append(val_loss / val_cnt)
        tic = time.time()
    
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
    
    train_file = r"D:\data\KG\train_triples_index.txt"
    val_file = r"D:\data\KG\val_triples_index.txt"

    train(model, train_file, val_file, lr, epochs)
