import random
from copy import deepcopy
import numpy as np

from model.config import LongTensor
from .config import SEP


def corrupt_triple(raw_batch: list, entityTotal: int):
    # repalce the head or tail with random function
    negs = deepcopy(raw_batch)
    if random.random() < 0.5:
        # replace the head
        while True:
            new_head = [random.randint(0, entityTotal - 1) for i in range(len(negs[0]))]
            if new_head != negs[0]:
                break
        negs[0] = new_head
    else:
        # replace the tail
        while True:
            new_tail = [random.randint(0, entityTotal - 1) for i in range(len(negs[2]))]
            if new_tail != negs[2]:
                break
        negs[2] = new_tail
    return negs


def load_batch_triple(filename: str, batch_size: int) -> tuple:
    """
    由于数据量太大，只好使用generator节约内存
    返回结果形如：(((ph),(pr),(pt)), ((nh),(nr),(nt)))
    """
    f = open(filename)
    toInt = lambda x: int(x)
    with open("note.log") as note:
        ent_cnt, _ = note.readlines()
        ent_cnt = int(ent_cnt.strip("\n").split(SEP)[1])

    pos = [[], [], []]
    for line in f:
        tps = line.rstrip("\n").split(SEP)
        h, r, t = tuple(map(toInt, tps))
        pos[0].append(h)
        pos[1].append(r)
        pos[2].append(t)
        if len(pos[0]) == batch_size:
            negs = corrupt_triple(pos, ent_cnt)
            yield LongTensor(pos), LongTensor(negs)
            pos = [[], [], []]
    if len(pos[0]) != 0:
        negs = corrupt_triple(pos, ent_cnt)
        yield LongTensor(pos), LongTensor(negs)
    f.close()
