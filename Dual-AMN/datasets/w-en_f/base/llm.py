from collections import defaultdict

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

def read_exp(file):
    exp = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        cur_exp = []
        for line in lines:
            cur = line.strip().split('\t')
            if cur[0] == '0':
                exp.append(cur_exp)
            else:
                cur_exp.append(cur[0])
    return exp

def read_llm_explanation(file):
    exp = []
    start = 0
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'triplets' in line or 'Triplets' in line:
                if start != 0:
                    exp.append(cur_exp)
                    # print(cur_exp)
                start = 1
                cur_exp = []
            else:
                if ' - ' in line:
                    if line[0] == '-':
                        line = line[1:]
                    cur = line.split(' - ')
                    tri1 = cur[0].strip()
                    tri1 = tri1.strip()
                    tri2 = cur[1].strip()
                    tri2 = tri2.strip()
                    index = tri1.find('(')
                    tri1 = tri1[index + 1: len(tri1) - 1]
                    index = tri2.find('(')
                    tri2 = tri2[index + 1: len(tri2) - 1]
                elif line == '':
                    continue
                else:
                    index = line.find(')')
                    index1 = line.find(')', index + 1)
                    if index == -1 or index1 == -1:
                        continue
                    print(line)
                    assert(0)
                tmp1 = tri1.split(',')
                tmp2 = tri2.split(',')
                t1 = []
                t2 = []
                for i in range(len(tmp1)):
                    cur = tmp1[i].strip()
                    cur = cur.replace(' ', '_')
                    t1.append(cur) 
                for i in range(len(tmp2)):
                    cur = tmp2[i].strip()
                    # cur = cur.replace(' ', '_')
                    t2.append(cur)
                cur_exp.append(t1)
                cur_exp.append(t2) 
        exp.append(cur_exp)
        # for cur in exp:
            # print(cur)
    with open('llm_exp', 'w', encoding='utf-8') as f:
        s = set()
        for cur in exp:
            for tri in cur:
                # print(tri)
                cur_exp = ''
                for i in range(len(tri) - 1):
                    cur_exp += tri[i] + ','
                cur_exp += tri[-1]
                # f.write(tri[0] + '\t' + tri[1] + '\t' + tri[2] + '\n')
                # f.write(e_dict_r[tri[0]] + '\t' + r_dict_r[tri[1]] + '\t' + e_dict_r[tri[2]] + '\n')
                if cur_exp not in s:
                    f.write(cur_exp + '\n')
                s.add(cur_exp)
            f.write(str(0) + '\t' + str(0) + '\t' + str(0) + '\n')


def read_llm_explanation_per(file):
    exp = []
    start = 0
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'Based on' in line:
                if start != 0:
                    exp.append(cur_exp)
                cur_exp = []
                start = 1
            elif '. f' in line:
                cur_exp.append(int(line.split('. f')[-1]))
        exp.append(cur_exp)
    return exp
                    
def read_llm_per_tri(file):
    tri_index = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            tri_list = line.split('\t')
            new_list = []
            for tri in tri_list:
                cur = tri.split(',')
                new_list.append(cur[0] + '\t' + cur[1] + '\t' + cur[2])
            tri_index.append(new_list)
    return tri_index

ori_pair = read_pair('pair.txt')
ground_pair = read_pair('test')
'''
tmp = ori_pair - (ori_pair & ground_pair)
for e1, e2 in tmp:
    print('{}\t{}'.format(e1, e2))
exit(0)
'''

repair_pair = read_pair('repair_pair')

# exp_pair  = read_pair_list('pair_llm')
'''
wrong_link, _ = read_link('wrong')
wrong =  ori_pair - (ori_pair & ground_pair) 
correct = ori_pair & ground_pair
correct_e = ori_pair & ground_pair & repair_pair
wrong_e = (repair_pair & ground_pair) - (ori_pair & ground_pair)
tmp = set()
for e1, e2 in wrong_e:
    tmp.add((e1, wrong_link[e1]))

for e1, e2 in tmp:
    print('{}\t{}'.format(e1, e2))
exit(0)
'''

repair = 1
llm_repair = 1
exp = 0
exp_per = 1
mix = 1
# print('wrong repair: ', len(ori_pair & ground_pair) - len(ori_pair & ground_pair & repair_pair))
# print('left wrong pair: ', len(ori_pair & repair_pair - ground_pair))
e_dict, e_dict_r = read_link('ent_dict_name')
r_dict, r_dict_r =  read_link('rel_dict_name')
# wrong =  (repair_pair & ground_pair) - (ground_pair & ori_pair)
# wrong = repair_pair & (ground_pair - (ground_pair & ori_pair))
# read_llm_explanation('exp_llm_raw')
if llm_repair == 1:
    llm_res = read_list('llm_repair_raw')
    res = []
    for i in range(len(llm_res)):
        cur = llm_res[i]
        if 'Incorrect' in cur or 'InCorrect' in cur or 'incorrect' in cur:
            res.append(0)
        elif 'correct' in cur or 'Correct' in cur:
            res.append(1)
        else:
            print(cur)
            assert(0)
    i = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    sample = read_pair_list('sample_1000')
    for e1, e2 in sample:
        if (e1, e2) in ground_pair:
            if res[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if res[i] == 1:
                FP += 1
            else:
                TN += 1
        i += 1
    print(TP, FN)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print(P)
    print(R)
    print('F1: {}'.format(2 * (P * R) / (P + R)) )
    
    tri1 = read_tri('triples_1')
    tri2 = read_tri('triples_2')
    exp_tri = defaultdict(set)
    sample = read_pair_list('sample_1000')
    repair_link = read_link('repair_pair')
    i = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for e1, e2 in sample:
        if (e1, e2) in ground_pair:
            if res[i] == 1:
                TP += 1
            elif (e1, e2) in repair_pair:
                TP += 1
            else:
                FN += 1
        else:
            if res[i] == 0:
                TN += 1
            elif (e1, e2) not in repair_pair:
                TN += 1
            else:
                FP += 1
        i += 1
    print(TP, FN, FP, TN)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print(P)
    print(R)
    print('F1: {}'.format(2 * (P * R) / (P + R)) )


elif repair == 1:
    # wrong =  repair_pair  - (ground_pair & repair_pair)
    '''
    w_2_c, _ = read_link('w_2_c')
    wrong = set()
    for e1, e2 in ori_pair:
        if e1 in w_2_c:
            wrong.add((e1, e2))
    '''
    '''
    print('Give you some entity alignment pairs, please determine which ones are wrong entity alignment pairs?')
    for cur in wrong:
        print('(' + e_dict[cur[0]].split('/')[-1] + ',' + e_dict[cur[1]].split('/')[-1] + ')' ) 
    '''
    
    tri1 = read_tri('triples_1')
    tri2 = read_tri('triples_2')
    exp_tri = defaultdict(set)
    sample = read_pair_list('sample_1000')
    # sample = read_pair_list('pair.txt')
    repair_link = read_link('repair_pair')
    '''
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for e1, e2 in sample:
        if (e1, e2) in ground_pair and (e1, e2) in repair_pair:
            TP += 1
        elif (e1, e2) in ground_pair and (e1, e2) not in repair_pair:
            FN += 1
        elif (e1, e2) not in ground_pair and (e1, e2) not in repair_pair:
            TN += 1
        elif (e1, e2) not in ground_pair and (e1, e2) in repair_pair:
            FP += 1
    print(TP, FN)
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    acc = (TP + TN) / (TP + FP + FN + TN)
    print(P)
    print(R)
    print(acc)
    print('F1: {}'.format(2 * (P * R) / (P + R)) )
    print(len(sample))
    '''
    
    for e1, e2 in sample:
        for cur in tri1[e1]:
            exp_tri[e_dict[e1]].add((e_dict[cur[0]].split('/')[-1], r_dict[cur[1]].split('/')[-1], e_dict[cur[2]].split('/')[-1]))
        for cur in tri2[e2]:
            exp_tri[e_dict[e2]].add((e_dict[cur[0]].split('/')[-1], r_dict[cur[1]].split('/')[-1], e_dict[cur[2]].split('/')[-1]))

    for e1, e2 in sample:
        print('Please check whether the claim is correct with the given evidence:')
        print('Claim: {} is same as {}'.format(e_dict[e1].split('/')[-1], e_dict[e2].split('/')[-1]))
        set1 = ''
        set2 = ''
        for tri in exp_tri[e_dict[e1]]:
            set1 += '{} {} {}, '.format(tri[0], tri[1], tri[2])
        for tri in exp_tri[e_dict[e2]]:
            set2 += '{} {} {}, '.format(tri[0], tri[1], tri[2])
        print('Evidence: {}'.format(set1 +  set2))
        print('Only output correct or incorrect to indicate whether the claim is correct or not')
        print('---------------------')
    
elif exp == 1:
    tri1 = read_tri('triples_1')
    tri2 = read_tri('triples_2')
    exp_tri = defaultdict(dict)
    for e1, e2 in exp_pair:
        for cur in tri1[e1]:
            exp_tri[e_dict[e1]][e_dict[cur[0]].split('/')[-1] + ','+ r_dict[cur[1]].split('/')[-1] + ','+ e_dict[cur[2]].split('/')[-1]] = cur
        for cur in tri2[e2]:
            exp_tri[e_dict[e2]][e_dict[cur[0]]+ ','+ r_dict[cur[1]] + ','+ e_dict[cur[2]]] = cur
    read_llm_explanation('llm_exp_raw')
    llm_exp = read_exp('llm_exp')
    # print(llm_exp)
    with open('exp_llm', 'w') as f:
        for i in range(len(exp_pair)):
            e1, e2 = exp_pair[i]
            cur_exp = llm_exp[i]
            print(exp_tri[e_dict[e1]])
            print(exp_tri[e_dict[e2]])
            print('-------------')
            for tri in cur_exp:
                if tri in exp_tri[e_dict[e1]]:
                    tmp = exp_tri[e_dict[e1]][tri]
                    f.write(tmp[0] + '\t' + tmp[1] + '\t' + tmp[2] + '\n')
                if tri in exp_tri[e_dict[e2]]:
                    tmp = exp_tri[e_dict[e2]][tri]
                    f.write(tmp[0] + '\t' + tmp[1] + '\t' + tmp[2] + '\n')
            f.write(str(0) + '\t' + str(0) + '\t' + str(0) + '\n')
                
elif exp_per ==  1:
    rank_index = read_llm_explanation_per('exp_llm_per_raw')
    tri = read_llm_per_tri('exp_llm_per_feature')
    '''
    i = 0
    for t in tri:
        print(i + 1, len(t))
        i += 1
    exit(0)
    '''
    with open('exp_llm_per', 'w') as f:
        for i in range(len(rank_index)):
            print(rank_index[i])
            if len(rank_index[i]) == 0:
                cur_list = tri[i]
                for cur in cur_list:
                    f.write(cur + '\n')
                f.write('0' + '\t' + '0' + '\t' + '0' + '\n')
            else:
                cur_list = tri[i]
                index = rank_index[i]
                for j in index:
                    print(i, j)
                    f.write(cur_list[j] + '\n')
                f.write('0' + '\t' + '0' + '\t' + '0' + '\n')

else:
    exp_pair  = read_pair_list('pair_llm')
    tri1 = read_tri('triples_1')
    tri2 = read_tri('triples_2')
    exp_tri = defaultdict(set)
    for e1, e2 in exp_pair:
        for cur in tri1[e1]:
            exp_tri[e_dict[e1]].add((e_dict[cur[0]].split('/')[-1], r_dict[cur[1]].split('/')[-1], e_dict[cur[2]].split('/')[-1]))
        for cur in tri2[e2]:
            exp_tri[e_dict[e2]].add((e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]))
    count = 0
    for e1, e2 in exp_pair:
        print('case: {}, {}'.format(e1, e2))
        print('Please identify the matching triplets in the following two triplet sets as explanation for predicting {} is same as {}:'.format(e_dict[e1].split('/')[-1], e_dict[e2].split('/')[-1]))
        set1 = ''
        set2 = ''
        for tri in exp_tri[e_dict[e1]]:
            set1 += '({},{},{}), '.format(tri[0], tri[1], tri[2])
        print('set1: {}'.format(set1))
        for tri in exp_tri[e_dict[e2]]:
            set2 += '({},{},{}), '.format(tri[0], tri[1], tri[2])
        print('set2: {}'.format(set2))
        print('example:')
        print('Please identify the matching triplets in the following two triplet sets as explanation for predicting 小明 is same as Xiaoming:')
        print('set1: (小明,父亲,小李), (小明,母亲,小红)')
        print('set2: (Xiaoming,father,Xiaoli), (Xiaoming,brother,Xiaozhang)')
        print('output: (小明,父亲,小李) - (Xiaoming,father,Xiaoli)')

        





