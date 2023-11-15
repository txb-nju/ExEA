def read_list(file):
    l = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip()
            l.append(cur)
    return l
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

def read_rel_link(file):
    d1 = {}
    d2 = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split(' ')
            tmp = ''
            for i in range(len(cur) - 1):
                tmp += cur[i + 1] + ' '
            d1[cur[0]] = tmp
            d2[tmp] = cur[0]
    return d1, d2

_, d1 = read_link('ent_links')
d2, _ = read_link('ill_ent_ids')
d3, d6 = read_link('ent_dict2')
d4, _ = read_link('D_name')

d5 = {}
d7, _ = read_rel_link('rel_name')
d8, d9 = read_link('rel_dict2')

d10 = {}
for cur in d9:
    if cur in d7:
        d10[int(d9[cur])] = d7[cur]
for i in range(248, 417):
    if i not in d10:
        print('{}\t{}'.format(i,d8[str(i)].split('/')[-1]))
    else:
        print('{}\t{}'.format(i,d10[i]))
'''
for cur in d2:
    d5[int(cur)] = d1[d3[d2[cur]]]

for i in range(15000):
    print('{}\t{}'.format(i,d5[i]))

d7 = {}
for cur in d6:
    if cur in d4:
        d7[int(d6[cur])] = d4[cur]
for i in range(15000, 30000):
    if i not in d7:
        print('{}\t{}'.format(i,d3[str(i)].split('/')[-1]))
    else:
        print('{}\t{}'.format(i,d7[i]))
'''


    