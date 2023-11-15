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

p1 = read_pair_list('pair_llm_m')
p2 = read_pair('pair.txt')
p3 = read_pair('pair_llm_m')
i = 0
for pair in p1:
    if pair in (p2 & p3):
        pass
    else:
        print(i)
    i += 1

