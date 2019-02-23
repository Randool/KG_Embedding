import torch


embed_dim = 4   # 嵌入维数
lr = 1e-3       # 学习率


# CUDA
CUDA = False # torch.cuda.is_available()

if CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
