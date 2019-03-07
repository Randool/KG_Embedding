import torch


embed_dim = 6   # 嵌入维数
lr = 5e-3       # 学习率
margin = 1.0    #
batch_size = 524288 # 2^19
epochs = 70
step_size = 15   # 学习率下调步长

# CUDA
CUDA = torch.cuda.is_available()

if CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
