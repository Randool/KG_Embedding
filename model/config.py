import torch


embed_dim = 4   # 嵌入维数
lr = 1e-3       # 学习率
margin = 1.0    #
batch_size = 524288 # 65535
epochs = 25
step_size = 5   # 学习率下调步长

# CUDA
CUDA = torch.cuda.is_available()

if CUDA:
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
