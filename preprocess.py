"""
数据预处理代码
"""
import os
from config import Dir, rawFile, encode, SEP, cookedFile


def data_pruge(r, t):
    """
    数据清洗，停用词、度量值、法律条款处理
    """
    if t[:3] == "<a>":
        t = t[3:-4]
    if r in rels:
        t = rels[r]
    if len(r) > 2 and r[0] == "第" and r[-1] == "条":
        t = "条款"
    return t


def convert_kg():
    """
    将实体/关系映射为数字，同时保存
        triples_index.txt   数字表示的三元组
        index_entity.txt    entity和数字的映射关系
        index_relation.txt  relation和数字的映射关系
    """
    print("Converting KG (id --> index)...")
    
    entity_id2index, relation_id2index = {}, {}
    entity_cnt, relation_cnt = 0, 0    

    file = open(rawFile, encoding=encode)
    # triples_index
    with open(os.path.join(Dir, "triples_index.txt"), "w") as f:
        for line in file.readlines():
            (h, r, t) = line.rstrip().split(SEP)
            # 数据清洗：去除DESC、停用词；度量值、法律条款处理
            if r == "DESC":
                continue
            t = data_pruge(r, t)
            if len(t) == 0:
                print(line)
                continue
            # 映射为数字
            if h not in entity_id2index:
                entity_id2index[h] = entity_cnt
                entity_cnt += 1
            if r not in relation_id2index:
                relation_id2index[r] = relation_cnt
                relation_cnt += 1
            if t not in entity_id2index:
                entity_id2index[t] = entity_cnt
                entity_cnt += 1

            triple = "{}\t{}\t{}\n".format(entity_id2index[h], relation_id2index[r], entity_id2index[t])
            f.write(triple)
        print("triples_index Done.")
    file.close()
    
    # 记录entity和relation总数，在产生neg数据时有用
    with open("note.log", "w") as f:
        info = "total_entity\t{}\ntotal_relation\t{}".format(entity_cnt, relation_cnt)
        print("'{}' has been writen to note.log".format(info))
        f.write(info)
    
    # index_entity
    filename = os.path.join(Dir, "index_entity.txt")
    with open(filename, "w", encoding=encode) as f:
        f.write(str(entity_id2index))
    print("index_entity done.")

    # index_relation
    filename = os.path.join(Dir, "index_relation.txt")
    with open(filename, "w", encoding=encode) as f:
        f.write(str(relation_id2index))
    print("index_relation done.")


def split_train_val_set(file: str, val_ratio=0.8):
    import numpy as np

    print("Loading {}".format(file))
    tps = np.loadtxt(file, dtype=np.int)
    print("Shuffling...")
    np.random.shuffle(tps)

    path, filename = os.path.split(file)
    num = int(len(tps) * val_ratio)
    
    print("Save to train data")
    with open(os.path.join(path, "train_" + filename), "w") as f:
        for i in range(num):
            f.write("{}\t{}\t{}\n".format(tps[i][0], tps[i][1], tps[i][2]))
    
    print("Save to val data")
    with open(os.path.join(path, "val_" + filename), "w") as f:
        for i in range(num, len(tps)):
            f.write("{}\t{}\t{}\n".format(tps[i][0], tps[i][1], tps[i][2]))


if __name__ == "__main__":
    rels = {}   # 如果不出现num_rels，定义rels可以防止convert_kg出现错误
    # 度量值 relation-tail 聚类
    if os.path.exists("num_rels.txt"):
        print("Loading numeric relations")
        with open("num_rels.txt", encoding=encode) as f:
            rels = f.read().split("\n")[:-1]
            rels = list(map(lambda x: x.split("\t"), rels))
            rels = dict(rels)   # 将二元组转化为字典
    
    # make KG
    convert_kg()

    # make train and validate set
    split_train_val_set(cookedFile)
