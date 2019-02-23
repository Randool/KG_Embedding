import os

# 文件
Dir = "D:\\data\\KG"
rawFile = os.path.join(Dir, "triples.txt")
cookedFile = os.path.join(Dir, "triples_index.txt")

encode = "utf-8"

## 分隔符
SEP = "\t"
## 停用词
stopwords = ["<a>", "</a>", "\n"]
