import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import embed_dim, FloatTensor


class TransE(nn.Module):
    def __init__(self, entity_cnt, relation_cnt, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ent_embed = nn.Embedding(entity_cnt, embed_dim)
        self.rel_embed = nn.Embedding(relation_cnt, embed_dim)
        # init
        self._init_weight(self.ent_embed, entity_cnt)
        self._init_weight(self.rel_embed, entity_cnt)

    def _init_weight(self, embedding, cnt):
        weight = FloatTensor(cnt, self.embed_dim)
        nn.init.xavier_normal_(weight)
        embedding.weight = nn.Parameter(weight)
        embedding.weight.data = F.normalize(embedding.weight.data, p=2, dim=1)

    def forward(self, ph, pr, pt, nh, nr, nt):
        # embedding lookup
        ph_e, pr_e, pt_e = self.ent_embed(ph), self.rel_embed(pr), self.ent_embed(pt)
        nh_e, nr_e, nt_e = self.ent_embed(nh), self.rel_embed(nr), self.ent_embed(nt)
        # calculate distance, L2
        pos = torch.sum((ph_e + pr_e - pt_e)**2, 1)
        neg = torch.sum((nh_e + nr_e - nt_e)**2, 1)
        return pos, neg
