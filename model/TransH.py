import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FloatTensor


class TransH(nn.Module):
    def __init__(self, entity_cnt, relation_cnt, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ent_embed = nn.Embedding(entity_cnt, embed_dim)
        self.rel_embed = nn.Embedding(relation_cnt, embed_dim)
        self.norm_embed = nn.Embedding(relation_cnt, embed_dim)
        # init
        self.init_weight(self.ent_embed, entity_cnt)
        self.init_weight(self.rel_embed, relation_cnt)
        self.init_weight(self.norm_embed, relation_cnt)
    
    def init_weight(self, embedding, cnt):
        weight = FloatTensor(cnt, self.embed_dim)
        nn.init.xavier_normal_(weight)
        embedding.weight = nn.Parameter(weight)
        embedding.weight.data = F.normalize(embedding.weight.data, p=2, dim=1)
    
    def projection(self, origin, norm):
        return origin - torch.sum(origin * norm, dim=1, keepdim=True) * norm
    
    def forward(self, ph, pr, pt, nh, nr, nt):
        # embedding lookup
        ph_e, pr_e, pt_e, p_norm = (
            self.ent_embed(ph), self.rel_embed(pr),
            self.ent_embed(pt), self.norm_embed(pr)
        )
        nh_e, nr_e, nt_e, n_norm = (
            self.ent_embed(nh), self.rel_embed(nr),
            self.ent_embed(nt), self.norm_embed(nr)
        )
        # map
        ph_e = self.projection(ph_e, p_norm)
        pt_e = self.projection(pt_e, p_norm)
        nh_e = self.projection(nh_e, n_norm)
        nt_e = self.projection(nt_e, n_norm)
        # calculate distance, L2
        pos = torch.sum((ph_e + pr_e - pt_e)**2, 1)
        neg = torch.sum((nh_e + nr_e - nt_e)**2, 1)
        return pos, neg