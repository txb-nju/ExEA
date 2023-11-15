import torch
from collections import defaultdict
from tqdm import trange

def read_link(file):
    d1 = {}
    d2 = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            d1[cur[0]] = cur[1]
            d2[cur[1]] = cur[0]
    return d1, d2

def read_pair(file):
    pair = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            pair.add((cur[0], cur[1]))
    return pair

def read_pair_list(file):
    pair = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            pair.append((cur[0], cur[1]))
    return pair

def read_list(file):
    l = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip()
            l.append(cur)
    return l

def read_pair_list(file):
    pair = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            pair.append((cur[0], cur[1]))
    return pair

def read_tri(file):
    tri = defaultdict(set)
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri[cur[0]].add((cur[0], cur[1], cur[2]))
            tri[cur[2]].add((cur[0], cur[1], cur[2]))
    return tri

def get_1_hop(e, tri):
    neigh1 = set()
    for cur in tri[e]:
        if cur[0] != e:
            neigh1.add(cur[0])
        else:
            neigh1.add(cur[2])
    return neigh1

def init_2_hop(e1, tri):
    neigh2 = set()
    neigh1 = get_1_hop(e1, tri)
    for ent in neigh1:
        neigh2 |= get_1_hop(ent, tri)
    if e1 in neigh2:
        neigh2.remove(e1)

    return neigh2 - neigh1 , neigh1

d1, d2 = read_link('ent_links')

tri1 = read_tri('triples_1')
tri2 = read_tri('triples_2')

pair = read_pair_list('ent_links')