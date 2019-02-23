import re
import os
import numpy as np
from config import Dir, rawFile, encode, SEP

"""
heads:1,857,754     relations:135,818        tails:12,718,355

[
    ('中华人民共和国合同法（法律）', 380), ('中华人民共和国证券法［已被修正］（法律）', 232), ('杭州工联大厦股份有限公司', 216),
    ('海南经济特区股份有限公司条例［修正］（法律）', 186), ('深圳经济特区股份有限公司条例［已被修正］（法律）', 179), ('中国证券交易系统有限公司业务规则（法律）', 175),
    ('市政府关于颁发《常州市行政执法程序暂行规定》的通知 （法律）', 175), ('中华人民共和国食品安全法(2015全文）（法律）', 170), 
    ('上海市股份有限公司暂行规定［失效］（法律）', 161), ('保险代理机构管理规定（法律）', 160)
]
[
    ('CATEGORY_ZH', 3270304), ('中文名', 1089688), ('DESC', 968843), ('国籍', 255581), ('歌手', 248855),
    ('长度', 243576), ('专辑', 243184), ('出生日期', 233904), ('周围景观', 226947), ('职业', 200549)
]
[
    ('人物', 335246), ('<a>中国</a>', 197326), ('地点', 169037), ('组织机构', 145678), ('<a>汉族</a>', 85977),
    ('字词', 83198), ('<a>男</a>', 77419), ('中国其他行政区划', 69635), ('游戏', 64987), ('村庄', 62190)
]
"""

def count_h_r_t():
    """
    heads:1,857,754   relations:135,818        tails:12,718,355
    """
    head_cnt, rel_cnt, tail_cnt = 0, 0, 0
    heads, relations, tails = {}, {}, {}

    with open(rawFile, encoding=encode) as fp:
        lines = map(lambda x: x.split(SEP), fp.readlines())
        for (h, r, t) in lines:
            if h not in heads:
                heads[h] = head_cnt
                head_cnt += 1
            if t not in tails:
                tails[h] = tail_cnt
                tail_cnt += 1
            if r not in relations:
                relations[r] = rel_cnt
                rel_cnt += 1

    print(f"heads:{head_cnt}\trelations:{rel_cnt}\ttails:{tail_cnt}")


def count_item():
    """ 显示head、relation、tail出现频率前5000的实体的频率分布 """
    tot_head = 1857754
    tot_rel = 135818
    tot_tails = 12718355

    import matplotlib.pyplot as plt
    import math

    heads, relations, tails = {}, {}, {}

    with open(rawFile, encoding=encode) as fp:
        lines = map(lambda x: x.strip().split(SEP), fp.readlines())
        print("Counting...")
        for (h, r, t) in lines:
            if h not in heads:
                heads[h] = 1
            else:
                heads[h] += 1
            if t not in tails:
                tails[t] = 1
            else:
                tails[t] += 1
            if r not in relations:
                relations[r] = 1
            else:
                relations[r] += 1

        print("Sorting...")
        func = lambda item: item[1]
        heads = sorted(heads.items(), key=func, reverse=True)[: int(tot_head * 0.001)]
        relations = sorted(relations.items(), key=func, reverse=True)[: int(tot_rel * 0.001)]
        tails = sorted(tails.items(), key=func, reverse=True)[: int(tot_tails * 0.001)]

        print(heads[:10])
        print(relations[:10])
        print(tails[:10])

        func = lambda item: math.log10(item[1])
        heads = list(map(func, heads))
        relations = list(map(func, relations))
        tails = list(map(func, tails))

        print("Drawing...")
        x = np.log10(np.arange(1, len(heads) + 1, 1))
        plt.plot(x, heads)
        plt.title("log10(head_0_001_freq)")
        plt.savefig("top_0_001_freq_head.jpg")
        plt.cla()

        x = np.log10(np.arange(1, len(relations) + 1, 1))
        plt.plot(x, relations)
        plt.title("log10(rel_0_001_freq)")
        plt.savefig("top_0_001_freq_rel.jpg")
        plt.cla()

        x = np.log10(np.arange(1, len(tails) + 1, 1))
        plt.plot(x, tails)
        plt.title("log10(tail_0_001_freq)")
        plt.savefig("top_0_001_freq_tail.jpg")


def which_is_complex(threshold=25):
    """ 显示最复杂的relation """
    cnt = {}
    with open(rawFile, encoding=encode) as f:
        for line in f.readlines():
            (_, r, t) = line.strip().split(SEP)
            t = t.lstrip("<a>").rstrip("</a>")

            if len(t) > threshold:
                if r not in cnt:
                    cnt[r] = 1
                else:
                    cnt[r] += 1
    
    complex_relations = sorted(cnt.items(), key=lambda x: x[1], reverse=True)[:100]
    print(complex_relations)


def numeric_relations():
    """ 找出和度量相关的relation """
    pat = re.compile(r"\d+")
    rels = {}
    with open(rawFile, encoding=encode) as f:
        for line in f.readlines():
            (_, r, t) = line.strip().split(SEP)
            if r == "DESC":
                continue
            if pat.search(t):
                if r not in rels:
                    rels[r] = 1
                else:
                    rels[r] += 1
    print("Scan done.")
    total = sum(map(lambda x: x[1], rels.items()))
    print(f"relations: {len(rels)}, tails: {total}")
    rels = sorted(rels.items(), key=lambda x: x[1], reverse=True)

    # 取相关tails总和超过总体tails的85%
    cnt, threshold = 0, int(total * 0.85)
    for (i, (_, num)) in enumerate(rels):
        cnt += num
        if cnt >= threshold:
            print(f"sum(Top-{i+1}) >= threshold!")
            break
    rels = rels[:i]
    with open("num_rels.txt", "w", encoding=encode) as f:
        rels = list(map(lambda x: x[0], rels))
        for rel in rels:
            f.write(f"{rel}\t{rel}\n")
    print("Write done.")


def legislation_relations():
    """ 找出和法律条款有关的数据 """
    with open(rawFile, encoding=encode) as f:
        for line in f.readlines():
            (_, r, t) = line.strip().split(SEP)
            if r == "DESC":
                continue
    pass


if __name__ == "__main__":
    # count_h_r_t()
    count_item()
    # which_is_complex()
    # numeric_relations()
    # legislation_relations()
