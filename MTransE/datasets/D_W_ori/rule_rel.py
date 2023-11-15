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

def read_r_func(file):
    r_func = {}
    r_func_r = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            r_func[cur[0]] = float(cur[1])
            r_func_r[cur[0]] = float(cur[2])
    return r_func, r_func_r

def read_rule(file):
    rule = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            rule.add((cur[0],cur[1],cur[2],cur[3]))
    return rule

rule = read_rule('triangle_id_1')
d1, d2 = read_link('rel_dict')
d3, d4 = read_link('rel_links')
r_func, r_func_r = read_r_func('r_func')
a = 1
for cur in rule:
    # print(cur)
    # print(r_func_r[cur[0]], r_func_r[cur[2]], r_func[cur[0]], r_func[cur[2]])
    if int(cur[1]) == 1 and r_func_r[cur[0]] >= a and int(cur[3]) == 1 and r_func_r[cur[2]] >= a:
        print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
    elif int(cur[1]) == 1 and r_func_r[cur[0]] >= a and int(cur[3]) == 0 and r_func[cur[2]] >= a:
        print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
    elif int(cur[1]) == 0 and r_func[cur[0]] >= a and int(cur[3]) == 0 and r_func[cur[2]] >= a:
        print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
    elif int(cur[1]) == 0 and r_func[cur[0]] >= a and int(cur[3]) == 1 and r_func_r[cur[2]] >= a:
        print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
 