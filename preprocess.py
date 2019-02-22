import os
import numpy as np
from config import Dir, rawFile, encode, SEP 


def data_purge(info: str) -> str:
    """ 数据清洗：去除停用词；判断该info是不是度量值 """
    if info[:3] == "<a>":
        info = info[3: -4]
    return info


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
            if r == "DESC":
                continue
            t = data_purge(t)
            if len(t) == 0:
                print(line)
                continue

            if h not in entity_id2index:
                entity_id2index[h] = entity_cnt
                entity_cnt += 1
            if r not in relation_id2index:
                relation_id2index[r] = relation_cnt
                relation_cnt += 1
            if t not in entity_id2index:
                entity_id2index[t] = entity_cnt
                entity_cnt += 1

            f.write(
                f"{entity_id2index[h]}\t{relation_id2index[r]}\t{entity_id2index[t]}\n"
            )
        print("triples_index Done.")
        print(f"number of entities: {entity_cnt}")
        print(f"number of relations: {relation_cnt}")
    file.close()
    
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


if __name__ == "__main__":
    convert_kg()
