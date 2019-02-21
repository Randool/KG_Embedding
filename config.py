import os

# 文件
Dir = "D:\\data\\KG"
rawFilename = "triples.txt"  # "triples.txt"
rawFile = os.path.join(Dir, rawFilename)

encode = "utf-8"

## 分隔符
SEP = "\t"
## 停用词
stopwords = ["<a>", "</a>", "\n"]

## 复杂语义关系
_relation = ["DESC", "酒店地址"]

# 嵌入维数
embed_dim = 4

# CUDA
import torch
CUDA = False # torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
LongTensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
