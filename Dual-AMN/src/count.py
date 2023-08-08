from collections import defaultdict
import numpy as np 
import matplotlib.pyplot as plt
import torch
import math
from itertools import combinations


def circle_graph(labels, sizes, colors, explode):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.figure(figsize=(7.5,5),dpi=80) #调节画布的大小
    # labels = ['雷雨','能见度','颠簸','积冰','冰雨','雪','低云','风'] #定义各个扇形的面积/标签
    # sizes = [24.24,17.17,7.7,7.7,8.8,9.9,14.14,14.14] #各个值，影响各个扇形的面积
    # colors = ['red','yellowgreen','lightskyblue','yellow','purple','pink','peachpuff','orange', 'blue', 'green'] #每块扇形的颜色
    # explode = (0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01,0.01)
    patches,text1,text2 = plt.pie(sizes,
                        explode=explode,
                        labels=labels,
                        colors=colors,
                        labeldistance = 1.2,#图例距圆心半径倍距离
                        autopct = '%3.2f%%', #数值保留固定小数位
                        shadow = False, #无阴影设置
                        startangle =90, #逆时针起始角度设置
                        pctdistance = 0.6) #数值距圆心半径倍数距离
    #patches饼图的返回值，texts1为饼图外label的文本，texts2为饼图内部文本
    plt.axis('equal')
    plt.legend()
    plt.show()

def neigh_sim():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    d1, d2 = read_link('zh-en/pair.txt')
    corr, _ = read_link('zh-en/all_links')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    two_hop1, _, _ = get_2_hop(tri1, None)
    two_hop2, _, _ = get_2_hop(tri2, None)
    nei1 = {}
    nei2 = {}
    for cur in one_hop1:
        nei1[cur] = one_hop1[cur] | two_hop1[cur]
    for cur in one_hop2:
        nei2[cur] = one_hop2[cur] | two_hop2[cur]
    sim_graph = defaultdict(int)
    sizes = [0,0,0]
    for cur in d1:
        if d1[cur] != corr[cur]:
            count = 0
            for neigh in nei1[cur]:
                if neigh in corr:
                    if corr[neigh] in nei2[corr[cur]]:
                        count += 1
            sim = count / (len(nei1[cur] | nei2[corr[cur]]) - count)
            if sim == 0:
                sizes[0] += 1
            elif sim <= 0.5:
                sizes[1] += 1
            else:
                sizes[2] += 1
            if sim <= 0.1:
                sim_graph[0.1] += 1
            elif sim <= 0.2:
                sim_graph[0.2] += 1
            elif sim <= 0.3:
                sim_graph[0.3] += 1
            elif sim <= 0.4:
                sim_graph[0.4] += 1
            elif sim <= 0.5:
                sim_graph[0.5] += 1
            elif sim <= 0.6:
                sim_graph[0.6] += 1
            elif sim <= 0.7:
                sim_graph[0.7] += 1
            elif sim <= 0.8:
                sim_graph[0.8] += 1
            elif sim <= 0.9:
                sim_graph[0.9] += 1
            elif sim <= 1:
                sim_graph[1] += 1
            # sim_graph[count / (len(one_hop1[cur] | one_hop2[d1[cur]]) - count)] += 1
    x = []
    y = []
    
    labels = ['no sim neigh', 'low sim neigh', 'high sim neigh']
    for cur in sim_graph:
        x.append(cur)
        # labels.append(str(cur))
        # sizes.append(sim_graph[cur])
        y.append(sim_graph[cur])
    # plt.scatter(x, y)
    # plt.show()
    colors = ['red','yellowgreen','lightskyblue'] #每块扇形的颜色
    explode = (0.01,0.01,0.01)
    # colors = ['red','yellowgreen','lightskyblue','yellow','purple','pink','peachpuff','orange', 'blue', 'green'] #每块扇形的颜色
    # explode = (0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01,0.01)
    circle_graph(labels, sizes, colors, explode)

def explain_neigh_sim0():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    d_c, _ = read_link('zh-en/all_links')
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    get_confidence_filter()
    r_align_0 = get_r_conf_all('zh-en/align_r_con_id_filter', 0)
   
    one_hop1, neigh_r1, _ = get_1_hop(tri1)
    one_hop2, neigh_r2, _ = get_1_hop(tri2)


    explain1 = defaultdict(set)
    explain2 = defaultdict(set)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e1 in one_hop1:
            for neigh in one_hop1[e1]:
                if neigh not in d1:
                    continue
                if e2 in one_hop2:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_0[(cur_r1, cur_r2)] > 0.1:
                                    explain1[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain2[(e1, e2)].add((e2, cur_r2, d1[neigh]))
    count_wrong = 0
    count_corr = 0
    no_explain = 0
    hit = 0
    explain = 0
    count = 0
    count_1_c = 0
    count_1_w = 0
    train = set()
    with open('zh-en/train_links', 'r',  encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            train.add((cur[0], cur[1]))
    with open('explain_wrong_explain', 'w', encoding='utf-8') as f, open('explain_correct_not', 'w', encoding='utf-8') as f1:
        for cur in d1:
            # print(ent[cur])
            e2 = d1[cur]
            e1 = cur
            if len(explain1[(e1, e2)]):
                if e2 == d_c[e1]:
                    count_corr += 1
                    s = set()
                    s2 = set()
                    flag = 0
                    flag2 = 0
                    for e in explain1[(e1, e2)]:
                        if len(s) == 0:
                            flag = 1
                            s = one_hop1[e[2]]
                        else:
                            flag = 0
                            s = one_hop1[e[2]] & s
                    for e in explain2[(e1, e2)]:
                        if len(s2) == 0:
                            flag2 = 1
                            s2 = one_hop2[e[2]]
                        else:
                            flag2 = 0
                            s2 = one_hop2[e[2]] & s2
                    if (len(s) > 1 and flag == 0) and (len(s2) > 1 and flag2 == 0) or len(explain1[(e1, e2)]) == 1:
                        # print(s)
                        
                        f1.write('---------------kg1--------------\n')
                        for cur in explain1[(e1, e2)]:
                            f1.write(ent[cur[0]] + '\t' + r[cur[1]] + '\t' + ent[cur[2]] + '\n')
                        f1.write('---------------kg2--------------\n')
                        for cur in explain2[(e1, e2)]:
                            f1.write(ent[cur[0]] + '\t' + r[cur[1]] + '\t' + ent[cur[2]] + '\n') 
                        
                        f1.write('---------------wrong kg1--------------\n')
                        for entity in s:
                            # print(ent[entity])
                            
                            if entity != e1:
                                for neigh in one_hop1[entity]:
                                    for cur_r in neigh_r1[(entity, neigh)]:
                                        f1.write(ent[entity]+ '\t' + r[cur_r]+ '\t' +  ent[neigh]+ '\n')
                        f1.write('---------------wrong kg2--------------\n')
                        for entity in s2:
                            # print(ent[entity])
                            
                            if entity != e2:
                                for neigh in one_hop2[entity]:
                                    for cur_r in neigh_r2[(entity, neigh)]:
                                        f1.write(ent[entity]+ '\t' + r[cur_r]+ '\t' +  ent[neigh]+ '\n')
                            
                        count_1_c += 1
                    else:
                        train.add((e1 ,e2))
                if e2 != d_c[e1]:
                    count_wrong += 1
                    s = set()
                    s2 = set()
                    flag = 0
                    flag2 = 0
                    for e in explain1[(e1, e2)]:
                        if len(s) == 0:
                            flag = 1
                            s = one_hop1[e[2]]
                        else:
                            flag = 0
                            s = one_hop1[e[2]] & s
                    for e in explain2[(e1, e2)]:
                        if len(s2) == 0:
                            flag2 = 1
                            s2 = one_hop2[e[2]]
                        else:
                            flag2 = 0
                            s2 = one_hop2[e[2]] & s2
                    if (len(s) > 1 and flag == 0) and (len(s2) > 1 and flag2 == 0) or len(explain1[(e1, e2)]) == 1:
                        count_1_w += 1
                    else:
                        train.add((e1 ,e2))
                    f.write('kg1:\n')
                    for cur in explain1[(e1, e2)]:
                        f.write(ent[cur[0]] + '\t' + r[cur[1]] + '\t' + ent[cur[2]] + '\n')
                    f.write('kg2:\n')
                    for cur in explain2[(e1, e2)]:
                        f.write(ent[cur[0]] + '\t' + r[cur[1]] + '\t' + ent[cur[2]] + '\n')
                    f.write('correct kg2:\n')
                    for neigh in one_hop2[d_c[e1]]:
                        for cur_r in neigh_r2[(d_c[e1], neigh)]:
                            f.write(ent[d_c[e1]]+ '\t' + r[cur_r]+ '\t' +  ent[neigh]+ '\n')
    
        print('1 in corr :', count_1_c / count_corr)
        print('1 in wrong :', count_1_w / count_wrong)
        with open('train_new', 'w', encoding='utf-8') as f:
            for cur in train:
                e1 = cur[0]
                e2 = cur[1]
                f.write(e1 + '\t' + e2 + '\n')
        '''
                print('kg1:')
                for neigh in one_hop1[e1]:
                    for cur_r in neigh_r1[(e1, neigh)]:
                        print(ent[e1], r[cur_r], ent[neigh])
                print('kg2:')
                for neigh in one_hop2[e2]:
                    for cur_r in neigh_r2[(e2, neigh)]:
                        print(ent[e2], r[cur_r], ent[neigh])
                print('correct kg2:')
                for neigh in one_hop2[d_c[e1]]:
                    for cur_r in neigh_r2[(d_c[e1], neigh)]:
                        print(ent[d_c[e1]], r[cur_r], ent[neigh])
        '''
        '''
    train = []
    for cur in d1:
        e2 = d1[cur]
        e1 = cur
        if len(explain1[(e1, e2)]):
            train.append([e1 ,e2])
    with open('train_new', 'w', encoding='utf-8') as f:
        for cur in train:
            e1 = cur[0]
            e2 = cur[1]
            f.write(e1 + '\t' + e2 + '\n')
        '''
        '''
        if len(explain1[(e1, e2)]):
            explain += 1
            if e2 == d_c[e1]:
                hit += 1
        '''
        '''
        if len(explain1[(e1, e2)]) == 0:
            # no_explain += 1
            if e2 == d_c[e1]:
                print('kg1:')
                for neigh in one_hop1[e1]:
                    for cur_r in neigh_r1[(e1, neigh)]:
                        print(ent[e1], r[cur_r], ent[neigh])
                print('kg2:')
                for neigh in one_hop2[e2]:
                    for cur_r in neigh_r2[(e2, neigh)]:
                        print(ent[e2], r[cur_r], ent[neigh])
        '''
    # print('hit by explain:', hit / explain)
    # print('hit by no_explain:', hit / no_explain)
    # print(count)

def explain_neigh_sim():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    d1, d2 = read_link('zh-en/dual_amn')
    d_c, _ = read_link('zh-en/all_links')
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    get_confidence_filter()
    r_align_0 = get_r_conf_all('zh-en/align_r_con_id_filter', 0)
   
    one_hop1, neigh_r1, _ = get_1_hop(tri1)
    one_hop2, neigh_r2, _ = get_1_hop(tri2)


    explain1 = defaultdict(set)
    explain2 = defaultdict(set)
    explain_score = defaultdict(float)
    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e1 in one_hop1:
            for neigh in one_hop1[e1]:
                if neigh not in d1:
                    continue
                if e2 in one_hop2:
                    if d1[neigh] in one_hop2[e2]:
                        count = 0
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_0[(cur_r1, cur_r2)] > 0.1:
                                    count += 1
                                    explain1[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain2[(e1, e2)].add((e2, cur_r2, d1[neigh]))
                                    explain_score[(e1, e2)] += r_align_0[(cur_r1, cur_r2)]
                        if count:
                            explain_score[(e1, e2)] /= count
    count_wrong = 0
    count_corr = 0
    no_explain = 0
    hit = 0
    explain = 0
    count = 0
    count_1_c = 0
    count_1_w = 0
    train = set()
    with open('zh-en/train_links', 'r',  encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            train.add((cur[0], cur[1]))
    with open('explain_wrong_explain', 'w', encoding='utf-8') as f:
        for cur in d1:
            e2 = d1[cur]
            e1 = cur
            if len(explain1[(e1, e2)]):
                
                s = set()
                s2 = set()
                flag = 0
                flag2 = 0
                kg1 = set()
                kg2 = set()
                for e in explain1[(e1, e2)]:
                    if len(s) == 0:
                        flag = 1
                        s = one_hop1[e[2]]
                    else:
                        flag = 0
                        s = one_hop1[e[2]] & s
                for e in explain2[(e1, e2)]:
                    if len(s2) == 0:
                        flag2 = 1
                        s2 = one_hop2[e[2]]
                    else:
                        flag2 = 0
                        s2 = one_hop2[e[2]] & s2
                if len(explain1[(e1, e2)]) == 1:
                    train.add((e1 ,e2))
                    if e2 != d_c[e1]:
                        count_1_w += 1
                        count_wrong += 1
                    else:
                        # for cur_r1 in neigh_r1[(, neigh)]:
                        # train.add((e1 ,e2))
                        count_1_c += 1
                        count_corr += 1
                else:
                    train.add((e1 ,e2))
                    if e2 != d_c[e1]:
                        count_wrong += 1
                    else:
                        count_corr += 1
                '''
                if flag == 1 and flag2 == 1:
                    # train.add((e1 ,e2))
                    if e2 != d_c[e1]:
                        count_1_w += 1
                        count_wrong += 1
                    else:
                        # for cur_r1 in neigh_r1[(, neigh)]:
                        # train.add((e1 ,e2))
                        count_1_c += 1
                        count_corr += 1
                    # continue
                    continue
                alpha = 0
                if flag == 0:
                    for entity in s:
                        if entity != e1:
                            count_all = 0
                            total = 0
                            for neigh in one_hop1[entity]:
                                if neigh in d1 and d1[neigh] in one_hop2[e2]:
                                    for cur_r1 in neigh_r1[(entity, neigh)]:
                                        for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                            if (cur_r1, cur_r2) in r_align_0:
                                                count_all += 1
                                                total += r_align_0[(cur_r1, cur_r2)]
                            if count_all:
                                total /= count_all
                            if total - explain_score[(e1, e2)] >= alpha and len(one_hop1[entity]) >len(one_hop1[e1]):
                                # print(total)
                                # print(explain_score[(e1, e2)])
                                flag = 1
                                if e2 != d_c[e1]:
                                    count_1_w += 1
                                    count_wrong += 1
                                else:
                                    count_1_c += 1
                                    count_corr += 1
                                continue
                                
                if flag2 == 0:
                    for entity in s2:
                        if entity != e2:
                            count_all = 0
                            total = 0
                            for neigh in one_hop2[entity]:
                                if neigh in d2 and d2[neigh] in one_hop1[e1]:
                                    for cur_r1 in neigh_r1[(entity, neigh)]:
                                        for cur_r2 in neigh_r1[(e1, d2[neigh])]:
                                            if (cur_r1, cur_r2) in r_align_0:
                                                count_all += 1
                                                total += r_align_0[(cur_r1, cur_r2)]
                            if count_all:
                                total /= count_all
                            if total - explain_score[(e1, e2)] >= alpha and len(one_hop2[entity]) > len(one_hop2[e2]):
                                # print(total)
                                # print(explain_score[(e1, e2)])
                                if e2 != d_c[e1]:
                                    count_1_w += 1
                                    count_wrong += 1
                                else:
                                    count_1_c += 1
                                    count_corr += 1
                                continue
                # count_corr += 1
                if e2 != d_c[e1]:
                    f.write('kg1-----------------------\n')
                    for cur in one_hop1[e1]:
                        for rel in neigh_r1[(e1, cur)]:
                            f.write(ent[e1]+ '\t' + r[rel]+ '\t' +  ent[cur]+ '\n')
                    f.write('kg2-----------------------\n')
                    for cur in one_hop2[e2]:
                        for rel in neigh_r2[(e2, cur)]:
                            f.write(ent[e2]+ '\t' + r[rel]+ '\t' +  ent[cur]+ '\n')
                    f.write('correct-------------------\n')
                    for cur in one_hop2[d_c[e1]]:
                        for rel in neigh_r2[(d_c[e1], cur)]:
                            f.write(ent[d_c[e1]]+ '\t' + r[rel]+ '\t' +  ent[cur]+ '\n')
                train.add((e1 ,e2))
                if e2 != d_c[e1]:
                    count_wrong += 1
                else:
                    count_corr += 1
                '''
                
            
                

    print('1 in corr :', count_1_c / count_corr)
    print('1 in wrong :', count_1_w / count_wrong)
    new_pair, new_dict = relvance()
    new_train = set()
    train_dict = {}
    for cur in train:
        train_dict[cur[0]] = cur[1]
    for cur in new_dict:
        new_train.add((cur, new_dict[cur]))
    
    for cur in train_dict:
        if cur not in new_dict:
            new_train.add((cur, train_dict[cur]))
    
    # new_train |= new_pair
    p = read_pair('zh-en/all_links')
    print(len(new_train & p) / len(p))
    print((len(new_train) - len(new_train & p)) / len(p))

    with open('train_new', 'w', encoding='utf-8') as f:
        for cur in new_train:
            e1 = cur[0]
            e2 = cur[1]
            f.write(e1 + '\t' + e2 + '\n')
        

def comp_sim():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    d1, d2 = read_link('zh-en/pair.txt')
    corr, _ = read_link('zh-en/all_links')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    two_hop1, _, _ = get_2_hop(tri1, None)
    two_hop2, _, _ = get_2_hop(tri2, None)
    nei1 = {}
    nei2 = {}
    for cur in one_hop1:
        nei1[cur] = one_hop1[cur] | two_hop1[cur]
    for cur in one_hop2:
        nei2[cur] = one_hop2[cur] | two_hop2[cur]
    sim_graph = defaultdict(int)
    sizes = [0,0]
    for cur in d1:
        count_cor = 0
        count_wro = 0
        neigh_len = len(nei1[cur])
        for other in d2:
            tmp = 0
            other_len = len(nei2[other])
            if other == corr[cur]:
                for neigh in nei1[cur]:
                    if neigh in corr:
                        if corr[neigh] in nei2[corr[cur]]:
                            count_cor += 1
                count_cor = count_cor / (neigh_len +  other_len - count_cor)
            else:
                for neigh in nei1[cur]:
                    if neigh in corr:
                        if corr[neigh] in nei2[other]:
                            tmp += 1
                count_wro = max(count_wro / (neigh_len +  other_len - count_wro),0)
        sim_diff = count_cor - count_wro
        if sim_diff <= 0:
            sizes[0] += 1
        else:
            sizes[1] += 1
            
    
    labels = ['wrong', 'correct']

    # plt.show()
    colors = ['red','yellowgreen'] #每块扇形的颜色
    explode = (0.01,0.01)
    # colors = ['red','yellowgreen','lightskyblue','yellow','purple','pink','peachpuff','orange', 'blue', 'green'] #每块扇形的颜色
    # explode = (0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01, 0.01,0.01)
    circle_graph(labels, sizes, colors, explode)

def count(file1, file2):
    s1 = set()
    s2 = set()
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
        
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
    
    print(len(s1 & s2) / len(s2))

def no_align(file1, file2):
    s1 = set()
    s2 = set()
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
        
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
    
    return s2 - (s1 & s2), s1 & s2

def wrong_align(file1, file2):
    s1 = set()
    s2 = set()
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
        
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
    
    return s1 - (s1 & s2), s1 & s2


def diff_align(file1, file2):
    s1 = set()
    s2 = set()
    d1 = {}
    d2 = {}
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
            d1[cur[0]] = cur[1]
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
            d2[cur[0]] = cur[1]
    return s1 - (s1 & s2), s1 & s2, d1, d2

def right_align(file1, file2):
    s1 = set()
    s2 = set()
    d1 = {}
    d2 = {}
    with open(file1) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s1.add((cur[0],cur[1]))
            d1[cur[0]] = cur[1]
    with open(file2) as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            s2.add((cur[0],cur[1]))
            d2[cur[0]] = cur[1]
    return s1 & s2, d1, d2


def read_tri_set(file):
    tri = defaultdict(set)
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri[cur[0]].add((cur[1], cur[2]))
    return tri
def read_tri(file):
    tri = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri.add((cur[0], cur[1], cur[2]))
    return tri

def read_tri_list(file):
    tri =[]
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            tri.append((cur[0], cur[1], cur[2]))
    return tri

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
def read_list(file):
    l = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            l.append((cur[0], cur[1]))
    return l
def get_diff():
    # count('zh-en/pair.txt', 'zh-en/all_links')
    tri1 = read_tri_set('zh-en/triples_1')
    tri2 = read_tri_set('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    # diffcult, corr = no_align('zh-en/pair.txt', 'zh-en/all_links')
    # diffcult, corr = wrong_align('zh-en/pair.txt', 'zh-en/all_links')
    diffcult, corr, d1, d2 = diff_align('zh-en/pair.txt', 'zh-en/all_links')
    ent1 = {}
    ent2 = {}
    for cur in corr:
        ent1[cur[0]] = cur[1]
        ent2[cur[1]] = cur[0]
    sign = 0
    for cur in diffcult:
        for tri in tri1[cur[0]]:
            if tri[1] in ent1:
                sign = ent[ent1[tri[1]]]
            else:
                sign = '0'
            print(r[tri[0]] + '\t' + ent[tri[1]] + '\t' + sign)
        print('wrong-----------------------')
        for tri in tri2[cur[1]]:
            if tri[1] in ent2:
                sign = ent[ent2[tri[1]]]
            else:
                sign = '0'
            print(r[tri[0]] + '\t' + ent[tri[1]] + '\t' + sign)
        print('correct-----------------------')
        for tri in tri2[d2[cur[0]]]:
            if tri[1] in ent2:
                sign = ent[ent2[tri[1]]]
            else:
                sign = '0'
            print(r[tri[0]] + '\t' + ent[tri[1]] + '\t' + sign)
        print('next-------------------------')


def get_r_conf(file):
    align_conf = defaultdict(int)
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            align_conf[(cur[0], cur[1])] = float(cur[2])
    return align_conf

def get_r_conf_all(file, select):
    align_conf = defaultdict(int)
    if select == 0:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                align_conf[(cur[0], cur[1])] = float(cur[2])
    elif select == 1:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                align_conf[cur[0], (cur[1], cur[2])] = float(cur[3])
    elif select == 2:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                align_conf[(cur[0], cur[1]), cur[2]] = float(cur[3])
    elif select == 3:
        with open(file) as f:
            lines = f.readlines()
            for line in lines:
                cur = line.strip().split('\t')
                align_conf[(cur[0], cur[1]), (cur[2], cur[3])] = float(cur[4])
    return align_conf
def get_right():
    # count('zh-en/pair.txt', 'zh-en/all_links')
    tri1 = read_tri_set('zh-en/triples_1')
    tri2 = read_tri_set('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    # diffcult, corr = no_align('zh-en/pair.txt', 'zh-en/all_links')
    # diffcult, corr = wrong_align('zh-en/pair.txt', 'zh-en/all_links')
    corr, d1, d2 = right_align('zh-en/pair.txt', 'zh-en/all_links')
    ent1 = {}
    ent2 = {}
    for cur in corr:
        ent1[cur[0]] = cur[1]
        ent2[cur[1]] = cur[0]
    sign = 0
    for cur in corr:
        for tri in tri1[cur[0]]:
            if tri[1] in ent1:
                sign = ent[ent1[tri[1]]]
            else:
                sign = '0'
            print(r[tri[0]] + '\t' + ent[tri[1]] + '\t' + sign)
        print('correct-----------------------')
        for tri in tri2[d2[cur[0]]]:
            if tri[1] in ent2:
                sign = ent[ent2[tri[1]]]
            else:
                sign = '0'
            print(r[tri[0]] + '\t' + ent[tri[1]] + '\t' + sign)
        print('next-------------------------')

def get_r_func_all():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    r1_func = defaultdict(int)
    r2_func = defaultdict(int)
    for cur in tri1:
        r1[cur[1]].add((cur[0], cur[2]))
    for cur in tri2:
        r2[cur[1]].add((cur[0], cur[2]))
    
    for cur in r1:
        x = defaultdict(int)
        for t in r1[cur]:
            x[t[0]] = 1
        r1_func[cur] = len(x) / len(r1[cur])
    
    for cur in r2:
        x = defaultdict(int)
        for t in r2[cur]:
            x[t[0]] = 1
        r2_func[cur] = len(x) / len(r2[cur])
    
    with open('zh-en/r_func1_id', 'w') as f:
        for cur in r1_func:
            f.write(cur + '\t' + str(r1_func[cur]) + '\n')
    with open('zh-en/r_func2_id', 'w') as f:
        for cur in r2_func:
            f.write(cur + '\t' + str(r2_func[cur]) + '\n')


def get_2_hop_direct(tri):
    one_hop = defaultdict(set)
    one_hop_inverse = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop_inverse[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for neigh2 in one_hop[neigh]:
                two_hop[cur].add(neigh2)
                for hop1 in neigh_r[(cur, neigh)]:
                    for hop2 in neigh_r[(neigh, neigh2)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
            if neigh not in one_hop_inverse:
                continue
            for neigh2 in one_hop_inverse[neigh]:
                two_hop[cur].add(neigh2)
                for hop1 in neigh_r[(cur, neigh)]:
                    for hop2 in neigh_r[(neigh2, neigh)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
    # print('len two hop:' ,len(two_hop))
    return two_hop, neigh_2_r, two_hop_r

def get_2_hop_no_mid(tri, r_func):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        neigh_r[(cur[2], cur[0])].add(cur[1])
    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for hop1 in neigh_r[(cur, neigh)]:
                # if r_func[hop1] > 0.5:
                    # continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
                    for hop2 in neigh_r[(neigh, neigh2)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
           
    # print('len two hop:' ,len(two_hop))
    return two_hop, neigh_2_r, two_hop_r

def get_2_hop(tri, r_func):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        neigh_r[(cur[2], cur[0])].add(cur[1])
    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for hop1 in neigh_r[(cur, neigh)]:
                # if r_func[hop1] > 0.5:
                    # continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
                    for hop2 in neigh_r[(neigh, neigh2)]:
                        neigh_2_r[(cur, neigh2)].add((hop1, neigh, hop2))
                        two_hop_r[(hop1, hop2)].add(cur)
           
    # print('len two hop:' ,len(two_hop))
    return two_hop, neigh_2_r, two_hop_r

def get_1_hop_direct(tri):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    one_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        one_hop_r[cur[1]].add(cur[0])
    return one_hop, neigh_r, one_hop_r


def get_1_hop(tri):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    one_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])
        neigh_r[(cur[0], cur[2])].add(cur[1])
        neigh_r[(cur[2], cur[0])].add(cur[1])
        one_hop_r[cur[1]].add(cur[0])
        one_hop_r[cur[1]].add(cur[2])
    return one_hop, neigh_r, one_hop_r


def get_confidence():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    neigh1 = defaultdict(set)
    neigh2 = defaultdict(set)
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    r_align = defaultdict(set)
    for cur in tri1:
        r1[cur[1]].add(cur[0])
        neigh1[cur[0]].add(cur[2])
        neigh_r1[(cur[0], cur[2])].add(cur[1])
    for cur in tri2:
        r2[cur[1]].add(cur[0])
        neigh2[cur[0]].add(cur[2])
        neigh_r2[(cur[0], cur[2])].add(cur[1])
    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in neigh2 or e1 not in neigh1:
            continue
        for neigh in neigh1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in neigh2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    r_con1 = defaultdict(int)
    r_con2 = defaultdict(int)
    r_con = defaultdict(int)

    for cur1 in r1:
        for cur2 in r2:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(r1[cur1]) , len(r2[cur2]))

    
    with open('zh-en/r_con_id', 'w') as f:
        for cur in r_con:
            f.write(cur[0] + '\t' + cur[1] + '\t' + str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_id', 'w') as f:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(cur[0] + '\t' + cur[1] + '\t' + str(r_con[cur]) + '\n')


def get_confidence_filter():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r1 = defaultdict(list)
    r2 = defaultdict(list)
    neigh1 = defaultdict(set)
    neigh2 = defaultdict(set)
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    r_align = defaultdict(set)
    for cur in tri1:
        r1[cur[1]].append(cur[0])
        r1[cur[1]].append(cur[2])
        neigh1[cur[0]].add(cur[2])
        neigh1[cur[2]].add(cur[0])
        neigh_r1[(cur[0], cur[2])].add(cur[1])
        neigh_r1[(cur[2], cur[0])].add(cur[1])
    for cur in tri2:
        r2[cur[1]].append(cur[0])
        r2[cur[1]].append(cur[2])
        neigh2[cur[0]].add(cur[2])
        neigh2[cur[2]].add(cur[0])
        neigh_r2[(cur[0], cur[2])].add(cur[1])
        neigh_r2[(cur[2], cur[0])].add(cur[1])
    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in neigh2 or e1 not in neigh1:
            continue
        for neigh in neigh1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in neigh2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    r_con1 = defaultdict(int)
    r_con2 = defaultdict(int)
    r_con = defaultdict(int)

    for cur1 in r1:
        for cur2 in r2:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(r1[cur1]) , len(r2[cur2]))

    
    with open('zh-en/r_con_filter', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0]] + '\t' + r[cur[1]] + '\t' + str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_filter', 'w') as f, open('zh-en/align_r_con_id_filter', 'w') as f1:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(r[cur[0]] + '\t' + r[cur[1]] + '\t' + str(r_con[cur]) + '\n')
                f1.write(cur[0] + '\t' + cur[1] + '\t' + str(r_con[cur]) + '\n')

def get_explain():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    neigh1 = defaultdict(set)
    neigh2 = defaultdict(set)
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    r_align = get_r_conf('zh-en/align_r_con_id')
    for cur in tri1:
        r1[cur[1]].add(cur[0])
        neigh1[cur[0]].add(cur[2])
        neigh_r1[(cur[0], cur[2])].add(cur[1])
    for cur in tri2:
        r2[cur[1]].add(cur[0])
        neigh2[cur[0]].add(cur[2])
        neigh_r2[(cur[0], cur[2])].add(cur[1])
    explain1 = defaultdict(set)
    explain2 = defaultdict(set)
    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in neigh2 or e1 not in neigh1:
            continue
        for neigh in neigh1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in neigh2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        if r_align[(cur_r1, cur_r2)] > 0.1:
                            explain1[(e1, e2)].add((e1, cur_r1, neigh))
                            explain2[(e1, e2)].add((e2, cur_r2, d1[neigh]))
    with open('zh-en/explain', 'w') as f:
        for cur in d1:
            e1 = cur
            e2 = d1[cur]
            f.write(ent[e1] + '\t' + ent[e2] + '\n')
            f.write('-------------explain------------------\n')
            if len(explain1[(e1, e2)]) == 0 and len(explain2[(e1, e2)]) == 0:
                f.write('no 1-hop explain\n')
                continue
            for exp in explain1[(e1, e2)]:
                f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
            for exp in explain2[(e1, e2)]:
                f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')

            f.write('-------------------------------------\n')
            

def get_explain_1(r1_func, r2_func):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    neigh1 = defaultdict(set)
    neigh2 = defaultdict(set)
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    
    r_align_0 = get_r_conf_all('zh-en/align_r_con_id_filter', 0)
    r_align_1 = get_r_conf_all('zh-en/align_r_con_1_id_filter', 2)
    r_align_2 = get_r_conf_all('zh-en/align_r_con_2_filter', 1)

    one_hop1, neigh_r1, _ = get_1_hop(tri1)
    one_hop2, neigh_r2, _ = get_1_hop(tri2)
    two_hop1, neigh_2_r1, _ = get_2_hop(tri1, r1_func)
    two_hop2, neigh_2_r2, _ = get_2_hop(tri2, r2_func)

    explain1 = defaultdict(set)
    explain2 = defaultdict(set)

    explain3 = defaultdict(set)
    explain4 = defaultdict(set)

    explain5 = defaultdict(set)
    explain6 = defaultdict(set)

    explain7 = defaultdict(set)
    explain8 = defaultdict(set)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e1 in one_hop1 and e2 in one_hop2:
            for neigh in one_hop1[e1]:
                if neigh in d1:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_0[(cur_r1, cur_r2)] > 0.1:
                                    explain1[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain2[(e1, e2)].add((e2, cur_r2, d1[neigh]))
        if e1 in one_hop1 and e2 in two_hop2:
            for neigh in one_hop1[e1]:
                if neigh in d1:
                    if d1[neigh] in two_hop2[e2]:
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_mid in neigh_2_r2[(e2, d1[neigh])]:
                                cur_r2 = (cur_mid[0], cur_mid[2])
                                if (cur_r1, cur_r2) in r_align_2:
                                    if r_align_2[(cur_r1, cur_r2)] > 0.1:
                                        explain3[(e1, e2)].add((e1, cur_r1, neigh))
                                        explain4[(e1, e2)].add((e2, cur_mid, d1[neigh]))
        if e1 in two_hop1 and e2 in one_hop2:
            for neigh in two_hop1[e1]:
                if neigh in d1:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_mid in neigh_2_r1[(e1, neigh)]:
                            cur_r1 = (cur_mid[0], cur_mid[2])
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if (cur_r1, cur_r2) in r_align_1:
                                    if r_align_1[(cur_r1, cur_r2)] > 0.1:
                                        explain5[(e1, e2)].add((e1, cur_mid, neigh))
                                        explain6[(e1, e2)].add((e2, cur_r2, d1[neigh]))

    with open('zh-en/explain_1_useful', 'w', encoding='utf-8') as f:
        for cur in d1:
            e1 = cur
            e2 = d1[cur]
            f.write(ent[e1] + '\t' + ent[e2] + '\n')
            f.write('-------------explain------------------\n')
            f.write('-------------1 to 2------------------\n')
            if len(explain1[(e1, e2)]) == 0:
                for exp in explain3[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
                for exp in explain4[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1][0]] + '\t' + ent[exp[1][1]] + '\t' + r[exp[1][2]] + '\t' + ent[exp[2]] + '\n')
                f.write('-------------2 to 1------------------\n')
                for exp in explain5[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1][0]] + '\t' + ent[exp[1][1]] + '\t' + r[exp[1][2]] + '\t' + ent[exp[2]] + '\n')
                for exp in explain6[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
            f.write('-------------------------------------\n')


def get_explain_all(r1_func, r2_func):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    neigh1 = defaultdict(set)
    neigh2 = defaultdict(set)
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    
    r_align_0 = get_r_conf_all('zh-en/align_r_con_id_filter', 0)
    r_align_1 = get_r_conf_all('zh-en/align_r_con_1_id_filter', 1)
    r_align_2 = get_r_conf_all('zh-en/align_r_con_2_filter', 2)
    r_align_3 = get_r_conf_all('zh-en/align_r_con_4_id_filter', 3)

    '''
    r_align_0 = get_r_conf_all('zh-en/align_r_con_id', 0)
    r_align_1 = get_r_conf_all('zh-en/align_r_con_1', 1)
    r_align_2 = get_r_conf_all('zh-en/align_r_con_2', 2)
    r_align_3 = get_r_conf_all('zh-en/align_r_con_id_4', 3)
    '''
    one_hop1, neigh_r1, _ = get_1_hop(tri1)
    one_hop2, neigh_r2, _ = get_1_hop(tri2)
    two_hop1, neigh_2_r1, _ = get_2_hop(tri1, r1_func)
    two_hop2, neigh_2_r2, _ = get_2_hop(tri2, r2_func)

    explain1 = defaultdict(set)
    explain2 = defaultdict(set)

    explain3 = defaultdict(set)
    explain4 = defaultdict(set)

    explain5 = defaultdict(set)
    explain6 = defaultdict(set)

    explain7 = defaultdict(set)
    explain8 = defaultdict(set)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e1 in one_hop1:
            for neigh in one_hop1[e1]:
                if neigh not in d1:
                    continue
                if e2 in one_hop2:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_0[(cur_r1, cur_r2)] > 0.1:
                                    explain1[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain2[(e1, e2)].add((e2, cur_r2, d1[neigh]))
                if e2 in two_hop2[e2]:
                    if d1[neigh] in two_hop2[e2]:
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_2_r2[(e2, d1[neigh])]:
                                if r_align_2[(cur_r1, cur_r2)] > 0.1:
                                    explain3[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain4[(e1, e2)].add((e2, cur_r2, d1[neigh]))
        if e1 in two_hop1:
            for neigh in two_hop1[e1]:
                if neigh not in d1:
                    continue
                if e2 in one_hop2:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_r1 in neigh_2_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_1[(cur_r1, cur_r2)] > 0.1:
                                    explain5[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain6[(e1, e2)].add((e2, cur_r2, d1[neigh]))
                if e2 in two_hop2:
                    if d1[neigh] in two_hop2[e2]:
                        for cur_r1 in neigh_2_r1[(e1, neigh)]:
                            for cur_r2 in neigh_2_r2[(e2, d1[neigh])]:
                                if r_align_3[(cur_r1, cur_r2)] > 0.1:
                                    explain7[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain8[(e1, e2)].add((e2, cur_r2, d1[neigh]))
    with open('zh-en/explain_compre_filter', 'w') as f:
        for cur in d1:
            e1 = cur
            e2 = d1[cur]
            f.write(ent[e1] + '\t' + ent[e2] + '\n')
            f.write('-------------explain------------------\n')
            if len(explain1[(e1, e2)]):            
                for exp in explain1[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
                for exp in explain2[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
            else:
                f.write('-------------1 to 2------------------\n')
                for exp in explain3[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
                for exp in explain4[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1][0]] + '\t' + r[exp[1][1]] + '\t' + ent[exp[2]] + '\n')
                f.write('-------------2 to 1------------------\n')
                for exp in explain5[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1][0]] + '\t' + r[exp[1][1]] + '\t' + ent[exp[2]] + '\n')
                for exp in explain6[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1]] + '\t' + ent[exp[2]] + '\n')
                f.write('-------------2 to 2------------------\n')
                for exp in explain7[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1][0]] + '\t' + r[exp[1][1]] + '\t' + ent[exp[2]] + '\n')
                for exp in explain8[(e1, e2)]:
                    f.write(ent[exp[0]] + '\t' + r[exp[1][0]] + '\t' + r[exp[1][1]] + '\t' + ent[exp[2]] + '\n')
            f.write('-------------------------------------\n')


def get_confidence_1():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r_align = defaultdict(set)
    one_hop, neigh_r, one_hop_r = get_1_hop(tri2)
    two_hop, neigh_2_r, two_hop_r = get_2_hop(tri1)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in one_hop or e1 not in two_hop:
            continue
        for neigh in two_hop[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in one_hop[e2]:
                for cur_r1 in neigh_2_r[(e1, neigh)]:
                    for cur_r2 in neigh_r[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    # print(len(r_align))
    r_con = defaultdict(int)

    for cur1 in two_hop_r:
        for cur2 in one_hop_r:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(two_hop_r[cur1]) , len(one_hop_r[cur2]))

    
    with open('zh-en/r_con_2_1_to_2', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + cur[1] + '\t' +  str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_1', 'w') as f:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(cur[0][0] + '\t' + cur[0][1] + '\t' + cur[1] + '\t' +  str(r_con[cur]) + '\n')



def get_confidence_1_filter(r1_func):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r_align = defaultdict(set)
    one_hop, neigh_r, one_hop_r = get_1_hop(tri2)
    two_hop, neigh_2_r, two_hop_r = get_2_hop(tri1, r1_func)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in one_hop or e1 not in two_hop:
            continue
        for neigh in two_hop[e1]:
            print(neigh)
            if neigh not in d1:
                # print('neigh not in align')
                continue
            # print('neigh align')
            if d1[neigh] in one_hop[e2]:
                for cur_r1 in neigh_2_r[(e1, neigh)]:
                    for cur_r2 in neigh_r[(e2, d1[neigh])]:
                        # print('get')
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    # print(len(r_align))
    r_con = defaultdict(int)

    for cur1 in two_hop_r:
        for cur2 in one_hop_r:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(two_hop_r[cur1]) , len(one_hop_r[cur2]))

    
    with open('zh-en/r_con_2_1_to_2_filter', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + r[cur[1]] + '\t' +  str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_1_filter', 'w') as f, open('zh-en/align_r_con_1_id_filter', 'w') as f1:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + r[cur[1]] + '\t' +  str(r_con[cur]) + '\n')
                f1.write(cur[0][0] + '\t' + cur[0][1] + '\t' + cur[1] + '\t' +  str(r_con[cur]) + '\n')

def get_confidence_2():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r_align = defaultdict(set)
    one_hop, neigh_r, one_hop_r = get_1_hop(tri1)
    two_hop, neigh_2_r, two_hop_r = get_2_hop(tri2)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in two_hop or e1 not in one_hop:
            continue
        for neigh in one_hop[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in two_hop[e2]:
                for cur_r1 in neigh_r[(e1, neigh)]:
                    for cur_r2 in neigh_2_r[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    # print(len(r_align))
    r_con = defaultdict(int)

    for cur1 in one_hop_r:
        for cur2 in two_hop_r:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(one_hop_r[cur1]) , len(two_hop_r[cur2]))

    
    with open('zh-en/r_con_2_2_to_1', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0]] + '\t' + r[cur[1][0]] + '\t' + r[cur[1][1]] + '\t' +  str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_2', 'w') as f:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(cur[0] + '\t' + cur[1][0] + '\t' + cur[1][1] + '\t' +  str(r_con[cur]) + '\n')

def get_confidence_2_filter(r2_func):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    r_align = defaultdict(set)
    one_hop, neigh_r, one_hop_r = get_1_hop(tri1)
    two_hop, neigh_2_r, two_hop_r = get_2_hop(tri2, r2_func)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in two_hop or e1 not in one_hop:
            continue
        for neigh in one_hop[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in two_hop[e2]:
                for cur_r1 in neigh_r[(e1, neigh)]:
                    for cur_r2 in neigh_2_r[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    # print(len(r_align))
    r_con = defaultdict(int)

    for cur1 in one_hop_r:
        for cur2 in two_hop_r:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(one_hop_r[cur1]) , len(two_hop_r[cur2]))

    
    with open('zh-en/r_con_2_2_to_1_filter', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0]] + '\t' + r[cur[1][0]] + '\t' + r[cur[1][1]] + '\t' +  str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_2_filter', 'w') as f:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(cur[0] + '\t' + cur[1][0] + '\t' + cur[1][1] + '\t' +  str(r_con[cur]) + '\n')

def get_confidence_3():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')

    r_align = defaultdict(set)
    neigh1, neigh_r1, r1 = get_2_hop(tri1)
    neigh2, neigh_r2, r2 = get_2_hop(tri2)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in neigh2 or e1 not in neigh1:
            continue
        for neigh in neigh1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in neigh2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    # print(len(r_align))
    r_con = defaultdict(int)

    for cur1 in r1:
        for cur2 in r2:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(r1[cur1]) , len(r2[cur2]))

    
    with open('zh-en/r_con_4', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + r[cur[1][0]] + '\t' + r[cur[1][1]] + '\t' + str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_4', 'w') as f:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + r[cur[1][0]] + '\t' + r[cur[1][1]] + '\t' + str(r_con[cur]) + '\n')


def get_confidence_3_filter(r1_func, r2_func):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    d1, d2 = read_link('zh-en/pair.txt')

    r_align = defaultdict(set)
    neigh1, neigh_r1, r1 = get_2_hop(tri1, r1_func)
    neigh2, neigh_r2, r2 = get_2_hop(tri2, r2_func)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in neigh2 or e1 not in neigh1:
            continue
        for neigh in neigh1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in neigh2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))
    # print(len(r_align))
    r_con = defaultdict(int)

    for cur1 in r1:
        for cur2 in r2:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(r1[cur1]) , len(r2[cur2]))

    
    with open('zh-en/r_con_4_filter', 'w') as f:
        for cur in r_con:
            f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + r[cur[1][0]] + '\t' + r[cur[1][1]] + '\t' + str(r_con[cur]) + '\n')

    with open('zh-en/align_r_con_4_filter', 'w') as f, open('zh-en/align_r_con_4_id_filter', 'w') as f1:
        for cur in r_con:
            if r_con[cur] > 0.1:
                f.write(r[cur[0][0]] + '\t' + r[cur[0][1]] + '\t' + r[cur[1][0]] + '\t' + r[cur[1][1]] + '\t' + str(r_con[cur]) + '\n')
                f1.write(cur[0][0] + '\t' + cur[0][1] + '\t' + cur[1][0] + '\t' + cur[1][1] + '\t' + str(r_con[cur]) + '\n')
def get_explain_count():
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    d1, d2 = read_link('zh-en/pair.txt')
    d_c, _ = read_link('zh-en/all_links')
    neigh_r1 = defaultdict(set)
    neigh_r2 = defaultdict(set)
    r_align_0 = get_r_conf_all('zh-en/align_r_con_id', 0)
    r_align_1 = get_r_conf_all('zh-en/align_r_con_1', 1)
    r_align_2 = get_r_conf_all('zh-en/align_r_con_2', 2)
    r_align_3 = get_r_conf_all('zh-en/align_r_con_id_4', 3)
    
    one_hop1, neigh_r1, _ = get_1_hop(tri1)
    one_hop2, neigh_r2, _ = get_1_hop(tri2)
    two_hop1, neigh_2_r1, _ = get_2_hop(tri1)
    two_hop2, neigh_2_r2, _ = get_2_hop(tri2)

    explain1 = defaultdict(set)
    explain2 = defaultdict(set)

    explain3 = defaultdict(set)
    explain4 = defaultdict(set)

    explain5 = defaultdict(set)
    explain6 = defaultdict(set)

    explain7 = defaultdict(set)
    explain8 = defaultdict(set)

    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e1 in one_hop1:
            for neigh in one_hop1[e1]:
                if neigh not in d1:
                    continue
                if e2 in one_hop2:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_0[(cur_r1, cur_r2)] > 0.1:
                                    explain1[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain2[(e1, e2)].add((e2, cur_r2, d1[neigh]))
                if e2 in two_hop2:
                    if d1[neigh] in two_hop2[e2]:
                        print('having 1 to 2')
                        for cur_r1 in neigh_r1[(e1, neigh)]:
                            for cur_r2 in neigh_2_r2[(e2, d1[neigh])]:
                                print('having 1 to 2 rule')
                                print(r_align_2[(cur_r1, cur_r2)])
                                if r_align_2[(cur_r1, cur_r2)] > 0.1:
                                    explain3[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain4[(e1, e2)].add((e2, cur_r2, d1[neigh]))
        if e1 in two_hop1:
            for neigh in two_hop1[e1]:
                if neigh not in d1:
                    continue
                if e2 in one_hop2:
                    if d1[neigh] in one_hop2[e2]:
                        for cur_r1 in neigh_2_r1[(e1, neigh)]:
                            for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                                if r_align_1[(cur_r1, cur_r2)] > 0.1:
                                    explain5[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain6[(e1, e2)].add((e2, cur_r2, d1[neigh]))
                if e2 in two_hop2:
                    if d1[neigh] in two_hop2[e2]:
                        for cur_r1 in neigh_2_r1[(e1, neigh)]:
                            for cur_r2 in neigh_2_r2[(e2, d1[neigh])]:
                                if r_align_3[(cur_r1, cur_r2)] > 0.1:
                                    explain7[(e1, e2)].add((e1, cur_r1, neigh))
                                    explain8[(e1, e2)].add((e2, cur_r2, d1[neigh]))
    count_tp = 0
    count_fp = 0
    count_fn = 0
    count_tn = 0
    count_tp1 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if len(explain1[(e1, e2)]):
            count1 += 1
        if len(explain3[(e1, e2)]):
            count2 += 1
        if len(explain5[(e1, e2)]):
            count3 += 1
        if len(explain7[(e1, e2)]):
            count4 += 1
        
        if e2 != d_c[e1]:
            if len(explain1[(e1, e2)]) or len(explain3[(e1, e2)]) or len(explain5[(e1, e2)]) or len(explain7[(e1, e2)]):            
                count_fp += 1
            else:
                count_tn += 1
        else:
            if len(explain1[(e1, e2)]) or len(explain3[(e1, e2)]) or len(explain5[(e1, e2)]) or len(explain7[(e1, e2)]):            
                count_tp += 1
                if len(explain1[(e1, e2)]):
                    count_tp1 += 1
            else:
                count_fn += 1
    print(count1)
    print(count2)
    print(count3)
    print(count4)
    print(count_tp1 / count_tp)
    print(count_tp / len(d1))
    print((count_tp + count_fp) / len(d1))
    pre = count_tp / (count_tp + count_fp)
    recall = count_tp / (count_tp + count_fn)
    print('precision : ', pre)
    print('recall : ', recall)
    print('f1 : ', 2 * (pre * recall) / (pre + recall))
            
def get_r_func(file):
    d = {}
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            d[cur[0]] = float(cur[1])
    return d

def get_entity_tri(label, select):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    d1, d2 = read_link('zh-en/all_links')
    
    if select == 0:
        tri = tri1
        d = d1
    else:
        tri = tri2
        d = d2
    r, _ = read_link('zh-en/rel_dict')
    ent, id = read_link('zh-en/ent_dict')
    tar = id[label]
    print('align is ', ent[d[tar]])
    for cur in tri:
        if cur[0] == tar or cur[2] == tar:
            print(ent[cur[0]], r[cur[1]], ent[cur[2]])


def write_neigh_sim(file_in, file_out):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    d1, d2 = read_link(file_in)
    one_hop1, _, _ = get_1_hop(tri1)
    one_hop2, _, _ = get_1_hop(tri2)
    neigh_sim = defaultdict(float)
    for cur in d1:
        count1 = 0
        for neigh in one_hop1[cur]:
            if neigh in d1:
                if d1[neigh] in one_hop2[d1[cur]]:
                    count1 += 1
        count2 = 0
        for neigh in one_hop2[d1[cur]]:
            if neigh in d2:
                if d2[neigh] in one_hop1[cur]:
                    count2 += 1
        count = min(count1, count2)
        neigh_sim[(cur, d1[cur])] = count / (len(one_hop1[cur]) + len(one_hop2[d1[cur]]) - count)

    with open(file_out, 'w', encoding='utf-8') as f:
        for cur in neigh_sim:
            f.write(cur[0] + '\t' + cur[1] + '\t' + str(neigh_sim[cur]) + '\n') 

def read_pair(file):
    p = set()
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            p.add((cur[0], cur[1]))
    return p

def baseline_tri():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop_direct(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop_direct(tri2)
    r_align = defaultdict(set)
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    for t in tri1:
        r1[t[1]].add(t[0])
        r1[t[1]].add(t[2])
    for t in tri2:
        r2[t[1]].add(t[0])
        r2[t[1]].add(t[2])
    for cur in d1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in one_hop2 or e1 not in one_hop1:
            continue
        for neigh in one_hop1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in one_hop2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        r_align[(cur_r1, cur_r2)].add((e1, e2))

    r_con1 = defaultdict(int)
    r_con2 = defaultdict(int)
    r_con = defaultdict(int)

    for cur1 in r1:
        for cur2 in r2:
            if (cur1, cur2) in r_align:
                r_con[(cur1, cur2)] = len(r_align[(cur1, cur2)]) / max(len(r1[cur1]) , len(r2[cur2]))
    
    f_tri1 = set()
    f_tri2 = set()
    file_sample = '/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair'
    d3, d4 = read_link(file_sample)
    test_e1 = set()
    test_e2 = set()
    for cur in d3:
        test_e1.add(cur)
        test_e2.add(d3[cur])
    candidate_tri1 = set()
    candidate_tri2 = set()
    for cur in tri1:
        if cur[0] not in test_e1 and cur[2] not in test_e1:
            f_tri1.add(cur)
        else:
            candidate_tri1.add(cur)
    for cur in tri2:
        if cur[0] not in test_e2 and cur[2] not in test_e2:
            f_tri2.add(cur)
        else:
            candidate_tri2.add(cur) 
    nec_tri1 = defaultdict(set)
    nec1 = set()
    nec2 = set()
    nec_tri2 = defaultdict(set)
    for cur in test_e1:
        e1 = cur
        e2 = d1[cur]
        if e2 not in one_hop2 or e1 not in one_hop1:
            continue
        for neigh in one_hop1[e1]:
            if neigh not in d1:
                continue
            if d1[neigh] in one_hop2[e2]:
                for cur_r1 in neigh_r1[(e1, neigh)]:
                    for cur_r2 in neigh_r2[(e2, d1[neigh])]:
                        if r_con[(cur_r1, cur_r2)] >= 0.1:
                            # print(r_con[(cur_r1, cur_r2)])
                            nec_tri1[e1].add((e1, cur_r1, neigh))
                            nec1.add((e1, cur_r1, neigh))
                            nec_tri2[e2].add((e2, cur_r2, d1[neigh]))
                            nec2.add((e2, cur_r2, d1[neigh]))
    print(len(candidate_tri1))
    print(len(nec1))
    candidate_tri1 -= nec1
    print(len(f_tri1))
    f_tri1 -= nec1
    print(len(candidate_tri1))
    candidate_tri2 -= nec2
    print(len(f_tri1))
    f_tri1 |= candidate_tri1
    print(len(f_tri1))
    f_tri2 |= candidate_tri2
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/nec_tri', 'w', encoding='utf-8') as f1:
        
        for cur in d3:
            f1.write('-------------kg1----------------------\n')
            for tri1 in nec_tri1[cur]:
                f1.write(ent[tri1[0]] + '\t' + r[tri1[1]] + '\t' + ent[tri1[2]] + '\n')
            f1.write('-------------kg2----------------------\n')
            for tri2 in nec_tri2[d3[cur]]:
                f1.write(ent[tri2[0]] + '\t' + r[tri2[1]] + '\t' + ent[tri2[2]] + '\n')
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/filter_tri1', 'w', encoding='utf-8') as f1, open('/data/xbtian/ContEA-main/datasets/zh-en_f/filter_tri2', 'w', encoding='utf-8') as f2:
        for tri1 in f_tri1:
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2] + '\n')
        for tri2 in f_tri2:
            f2.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2] + '\n')


def baseline_tri_align():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    r_align = defaultdict(set)
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    
    f_tri1 = set()
    f_tri2 = set()
    file_sample = '/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair'
    d3, d4 = read_link(file_sample)
    test_e1 = set()
    test_e2 = set()
    for cur in d3:
        test_e1.add(cur)
        test_e2.add(d3[cur])
    candidate_tri1 = set()
    candidate_tri2 = set()
    can1 = defaultdict(set)
    can2 = defaultdict(set)
    for cur in tri1:
        if cur[0] not in test_e1 and cur[2] not in test_e1:
            f_tri1.add(cur)
        else:
            candidate_tri1.add(cur)
        if cur[0] in test_e1:
            can1[cur[0]].add(cur)
        if cur[2] in test_e1:
            can1[cur[2]].add(cur)
    for cur in tri2:
        if cur[0] not in test_e2 and cur[2] not in test_e2:
            f_tri2.add(cur)
        else:
            candidate_tri2.add(cur) 
        if cur[0] in test_e2:
            can2[cur[0]].add(cur)
        if cur[2] in test_e2:
            can2[cur[2]].add(cur)
    nec_tri1 = defaultdict(set)
    nec1 = set()
    nec2 = set()
    nec_tri2 = defaultdict(set)
    for cur in test_e1:
        e1 = cur
        e2 = d1[cur]
        for tri in can1[e1]:
            if tri[0] in d1 and tri[2] in d1:
                if tri[0] == e1:
                    if d1[tri[2]] in one_hop2[e2]:
                        nec1.add(tri)
                        nec_tri1[e1].add(tri)
                else:
                    if d1[tri[0]] in one_hop2[e2]:
                        nec1.add(tri)
                        nec_tri1[e1].add(tri)
        for tri in can2[e2]:
            if tri[0] in d2 and tri[2] in d2:
                if tri[0] == e2:
                    if d2[tri[2]] in one_hop1[e1]:
                        nec2.add(tri)
                        nec_tri2[e2].add(tri)
                else:
                    if d2[tri[0]] in one_hop1[e1]:
                        nec2.add(tri)
                        nec_tri2[e2].add(tri)
        
    print(len(candidate_tri1))
    print(len(nec1))
    candidate_tri1 -= nec1
    print(len(f_tri1))
    f_tri1 -= nec1
    print(len(candidate_tri1))
    candidate_tri2 -= nec2
    print(len(f_tri1))
    f_tri1 |= candidate_tri1
    print(len(f_tri1))
    f_tri2 |= candidate_tri2
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/nec_tri', 'w', encoding='utf-8') as f1:
        
        for cur in d3:
            f1.write('-------------kg1----------------------\n')
            for tri1 in nec_tri1[cur]:
                f1.write(ent[tri1[0]] + '\t' + r[tri1[1]] + '\t' + ent[tri1[2]] + '\n')
            f1.write('-------------kg2----------------------\n')
            for tri2 in nec_tri2[d3[cur]]:
                f1.write(ent[tri2[0]] + '\t' + r[tri2[1]] + '\t' + ent[tri2[2]] + '\n')
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/filter_tri1', 'w', encoding='utf-8') as f1, open('/data/xbtian/ContEA-main/datasets/zh-en_f/filter_tri2', 'w', encoding='utf-8') as f2:
        for tri1 in f_tri1:
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2] + '\n')
        for tri2 in f_tri2:
            f2.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2] + '\n')

def get_2_hop_all(tri):
    one_hop = defaultdict(set)
    neigh_r = defaultdict(set)
    two_hop = defaultdict(set)
    neigh_2_r = defaultdict(set)
    two_hop_r = defaultdict(set)
    for cur in tri:
        one_hop[cur[0]].add(cur[2])
        one_hop[cur[2]].add(cur[0])

    for cur in one_hop:
        for neigh in one_hop[cur]:
            if neigh not in one_hop:
                continue
            for hop1 in neigh_r[(cur, neigh)]:
                # if r_func[hop1] > 0.5:
                    # continue
                for neigh2 in one_hop[neigh]:
                    if cur == neigh2:
                        continue
                    two_hop[cur].add(neigh2)
        two_hop[cur] |= one_hop[cur]
           
    # print('len two hop:' ,len(two_hop))
    return two_hop


def baseline_tri_2_hop():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    two_hop1 = get_2_hop_all(tri1)
    two_hop2 = get_2_hop_all(tri2)
    r_align = defaultdict(set)
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    
    f_tri1 = set()
    f_tri2 = set()
    file_sample = '/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair'
    d3, d4 = read_link(file_sample)
    test_e1 = set()
    test_e2 = set()
    for cur in d3:
        test_e1.add(cur)
        test_e2.add(d3[cur])
    candidate_tri1 = set()
    candidate_tri2 = set()
    can1 = defaultdict(set)
    can2 = defaultdict(set)
    for cur in tri1:
        if cur[0] not in test_e1 and cur[2] not in test_e1:
            f_tri1.add(cur)
        else:
            candidate_tri1.add(cur)
        if cur[0] in test_e1:
            can1[cur[0]].add(cur)
        if cur[2] in test_e1:
            can1[cur[2]].add(cur)
    for cur in tri2:
        if cur[0] not in test_e2 and cur[2] not in test_e2:
            f_tri2.add(cur)
        else:
            candidate_tri2.add(cur) 
        if cur[0] in test_e2:
            can2[cur[0]].add(cur)
        if cur[2] in test_e2:
            can2[cur[2]].add(cur)
    nec_tri1 = defaultdict(set)
    nec1 = set()
    nec2 = set()
    nec_tri2 = defaultdict(set)
    for cur in test_e1:
        e1 = cur
        e2 = d1[cur]
        for tri in can1[e1]:
            if tri[0] == e1:
                flag = 0
                if tri[2] in d1:
                    if d1[tri[2]] in two_hop2[e2]:
                        nec1.add(tri)
                        nec_tri1[e1].add(tri)
                        flag = 1
                if flag == 0:
                    for neigh in one_hop1[tri[2]]:
                        if neigh in one_hop1 and neigh != e1 and neigh in d1:
                            if d1[neigh] in two_hop2[e2]:
                                nec1.add(tri)
                                nec_tri1[e1].add(tri)
                                break
            else:
                flag = 0
                if tri[0] in d1:
                    if d1[tri[0]] in two_hop2[e2]:
                        nec1.add(tri)
                        nec_tri1[e1].add(tri)
                        flag = 1
                if flag == 0:
                    for neigh in one_hop1[tri[0]]:
                        if neigh in one_hop1 and neigh != e1 and neigh in d1:
                            if d1[neigh] in two_hop2[e2]:
                                nec1.add(tri)
                                nec_tri1[e1].add(tri)
                                break
        for tri in can2[e2]:
            if tri[0] == e2:
                flag = 0
                if tri[2] in d2:
                    if d2[tri[2]] in two_hop1[e1]:
                        nec2.add(tri)
                        nec_tri2[e2].add(tri)
                        flag = 1
                if flag == 0:
                    for neigh in one_hop2[tri[2]]:
                        if neigh in one_hop2 and neigh != e2 and neigh in d2:
                            if d2[neigh] in two_hop1[e1]:
                                nec2.add(tri)
                                nec_tri2[e2].add(tri)
                                break
            else:
                flag = 0
                if tri[0] in d2:
                    if d2[tri[0]] in two_hop1[e1]:
                        nec2.add(tri)
                        nec_tri2[e2].add(tri)
                        flag = 1
                
                if flag == 0:
                    for neigh in one_hop2[tri[0]]:
                        if neigh in one_hop2 and neigh != e2 and neigh in d2:
                            if d2[neigh] in two_hop1[e1]:
                                nec2.add(tri)
                                nec_tri2[e2].add(tri)
                                break
        
    print(len(candidate_tri1))
    print(len(nec1))
    candidate_tri1 -= nec1
    print(len(f_tri1))
    f_tri1 -= nec1
    print(len(candidate_tri1))
    candidate_tri2 -= nec2
    print(len(f_tri1))
    f_tri1 |= candidate_tri1
    print(len(f_tri1))
    f_tri2 |= candidate_tri2
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/nec_tri', 'w', encoding='utf-8') as f1:
        
        for cur in d3:
            f1.write('-------------kg1----------------------\n')
            for tri1 in nec_tri1[cur]:
                f1.write(ent[tri1[0]] + '\t' + r[tri1[1]] + '\t' + ent[tri1[2]] + '\n')
            f1.write('-------------kg2----------------------\n')
            for tri2 in nec_tri2[d3[cur]]:
                f1.write(ent[tri2[0]] + '\t' + r[tri2[1]] + '\t' + ent[tri2[2]] + '\n')
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/filter_tri1', 'w', encoding='utf-8') as f1, open('/data/xbtian/ContEA-main/datasets/zh-en_f/filter_tri2', 'w', encoding='utf-8') as f2:
        for tri1 in f_tri1:
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2] + '\n')
        for tri2 in f_tri2:
            f2.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2] + '\n')

def baseline_tri_sim():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    sim = torch.load('/data/xbtian/amie-master/zh-en/sim_cos.pt')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    r_align = defaultdict(set)
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    
    f_tri1 = set()
    f_tri2 = set()
    file_sample = '/data/xbtian/ContEA-explain/datasets/zh-en_f/base/sample_pair'
    d3, d4 = read_link(file_sample)
    test_e1 = set()
    test_e2 = set()
    for cur in d3:
        test_e1.add(cur)
        test_e2.add(d3[cur])
    candidate_tri1 = set()
    candidate_tri2 = set()
    can1 = defaultdict(set)
    can2 = defaultdict(set)
    for cur in tri1:
        if cur[0] not in test_e1 and cur[2] not in test_e1:
            f_tri1.add(cur)
        else:
            candidate_tri1.add(cur)
        if cur[0] in test_e1:
            can1[cur[0]].add(cur)
        if cur[2] in test_e1:
            can1[cur[2]].add(cur)
    for cur in tri2:
        if cur[0] not in test_e2 and cur[2] not in test_e2:
            f_tri2.add(cur)
        else:
            candidate_tri2.add(cur) 
        if cur[0] in test_e2:
            can2[cur[0]].add(cur)
        if cur[2] in test_e2:
            can2[cur[2]].add(cur)
    nec_tri1 = defaultdict(set)
    nec1 = set()
    nec2 = set()
    nec_tri2 = defaultdict(set)
    thred = 0.9
    for cur in test_e1:
        e1 = cur
        e2 = d1[cur]
        for tri in can1[e1]:
            if tri[0] == e1:
                for neigh in one_hop2[e2]:
                    if sim[int(tri[2]), int(neigh) - 19388] >= thred:
                        nec1.add(tri)
                        nec_tri1[e1].add(tri)
                        break
            else:
                for neigh in one_hop2[e2]:
                    if sim[int(tri[0]), int(neigh) - 19388] >= thred:
                        nec1.add(tri)
                        nec_tri1[e1].add(tri)
                        break
        for tri in can2[e2]:
            if tri[0] == e2:
                for neigh in one_hop1[e1]:
                    if sim[int(neigh), int(tri[2]) - 19388] >= thred:
                        nec2.add(tri)
                        nec_tri2[e2].add(tri)
                        break
            else:
                for neigh in one_hop1[e1]:
                    if sim[int(neigh), int(tri[0]) - 19388] >= thred:
                        nec2.add(tri)
                        nec_tri2[e2].add(tri)
                        break
        
    print(len(candidate_tri1))
    print(len(nec1))
    candidate_tri1 -= nec1
    print(len(f_tri1))
    f_tri1 -= nec1
    print(len(candidate_tri1))
    candidate_tri2 -= nec2
    print(len(f_tri1))
    f_tri1 |= candidate_tri1
    print(len(f_tri1))
    f_tri2 |= candidate_tri2
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/nec_tri', 'w', encoding='utf-8') as f1:
        
        for cur in d3:
            f1.write('-------------kg1----------------------\n')
            for tri1 in nec_tri1[cur]:
                f1.write(ent[tri1[0]] + '\t' + r[tri1[1]] + '\t' + ent[tri1[2]] + '\n')
            f1.write('-------------kg2----------------------\n')
            for tri2 in nec_tri2[d3[cur]]:
                f1.write(ent[tri2[0]] + '\t' + r[tri2[1]] + '\t' + ent[tri2[2]] + '\n')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/filter_tri1', 'w', encoding='utf-8') as f1, open('/data/xbtian/ContEA-explain/datasets/zh-en_f/filter_tri2', 'w', encoding='utf-8') as f2:
        for tri1 in f_tri1:
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2] + '\n')
        for tri2 in f_tri2:
            f2.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2] + '\n')

def get_rfunc():
    r1 = {}
    r2 = {}
    with open('zh-en/r_func1_id') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            r1[line[0]] = float(line[1])
    with open('zh-en/r_func2_id') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            r2[line[0]] = float(line[1])
    return r1, r2

def count_rel():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    r1, r2 = get_rfunc()
    r, _ = read_link('zh-en/rel_dict')
    r_1 = defaultdict(set)
    r_2 = defaultdict(set)
    coo = defaultdict(int)
    '''
    for cur in d1:
        e2 = d1[cur]
        for neigh in one_hop1[cur]:
            for cur_r1 in neigh_r1[(cur, neigh)]:
                r_1[cur_r1].add((cur, neigh))
                for neigh2 in one_hop2[e2]:
                    for cur_r2 in neigh_r2[(e2, neigh2)]:
                        r_2[cur_r2].add((e2, neigh2))
                        coo[(cur_r1, cur_r2)] += 1
    '''
    for cur in d1:
        e2 = d1[cur]
        for neigh in one_hop1[cur]:
            if neigh in d1 and d1[neigh] in one_hop2[e2]:
                for cur_r1 in neigh_r1[(cur, neigh)]:
                    r_1[cur_r1].add((cur, neigh))
                    neigh2 = d1[neigh]
                    for cur_r2 in neigh_r2[(e2, neigh2)]:
                        r_2[cur_r2].add((e2, neigh2))
                        coo[(cur_r1, cur_r2)] += 1
    r_rel = defaultdict(float)
    with open('zh-en/r_rel_align', 'w') as f,open('zh-en/r_rel_align_id', 'w') as f1:
        for cur in coo:
            r_rel[cur] = r1[cur[0]] * r2[cur[1]] * coo[cur] / (len(r_1[cur[0]]) + len(r_2[cur[1]]))
            #   if r_rel[cur] > 0.1:
            f.write(r[cur[0]] + '\t' + r[cur[1]] + '\t' + str(r_rel[cur]) + '\n')
            f1.write(cur[0] + '\t' + cur[1] + '\t' + str(r_rel[cur]) + '\n')

def load_r_rel():
    r_rel = defaultdict(float)
    with open('zh-en/r_rel_align_id', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            r_rel[(line[0], line[1])] = float(line[2])
    return r_rel

def change_tar(id):
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    sim = torch.load('/data/xbtian/amie-master/zh-en/sim_cos.pt')
    r_rel = load_r_rel()
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    r_align = defaultdict(set)
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    
    f_tri1 = set()
    f_tri2 = set()
    
    
    candidate_tri1 = set()
    candidate_tri2 = set()
    can1 = defaultdict(set)
    can2 = defaultdict(set)
    e2 = d1[id]
    thred = 0.9
    for cur in tri1:
        if cur[0] == id or cur[2] == id:
            candidate_tri1.add(cur)
        else:
            f_tri1.add(cur)
    for cur in tri2:
        if cur[0] == e2 or cur[2] == e2:
            candidate_tri2.add(cur)
        else:
            f_tri2.add(cur)
    nec1 = set()
    nec2 = set()
    for t1 in candidate_tri1:
        for t2 in candidate_tri2:
            if t1[0] == id:
                neigh1 = t1[2]
            else:
                neigh1 = t1[0]
            if t2[0] == e2:
                neigh2 = t2[2]
            else:
                neigh2 = t2[0]
            sim_e = sim[int(neigh1), int(neigh2) - 19388]
            rel_r = r_rel[(t1[1], t2[1])]
            rel_fact = sim_e + rel_r
            print(rel_fact)
            
            if rel_fact >= 0.85:
                nec1.add(t1)
                nec2.add(t2)
            

    '''
    thred = 0.9
    for cur in tri1:

        if cur[0] == id:
            neigh = cur[2]
            for n in one_hop2[e2]:
                if sim[int(neigh), int(n) - 19388] >= thred:
                    candidate_tri1.add(cur)
        elif cur[2] == id:
            neigh = cur[0]
            neigh = cur[2]
            for n in one_hop2[e2]:
                if sim[int(neigh), int(n) - 19388] >= thred:
                    candidate_tri1.add(cur)
        else:
            f_tri1.add(cur)
    for cur in tri2:
        if cur[0] == e2:
            neigh = cur[2]
            for n in one_hop1[id]:
                if sim[int(n), int(neigh) - 19388] >= thred:
                    candidate_tri2.add(cur)
        elif cur[2] == e2:
            neigh = cur[0]
            for n in one_hop1[id]:
                if sim[int(n), int(neigh) - 19388] >= thred:
                    candidate_tri2.add(cur)
        else:
            f_tri2.add(cur)
    
    for cur in tri1:
        if cur[0] == id:
            neigh = cur[2]
            if neigh in d1 and d1[neigh] in one_hop2[e2]:
                candidate_tri1.add(cur)
        elif cur[2] == id:
            neigh = cur[0]
            if neigh in d1 and d1[neigh] in one_hop2[e2]:
                candidate_tri1.add(cur)
        else:
            f_tri1.add(cur)
    for cur in tri2:
        if cur[0] == e2:
            neigh = cur[2]
            if neigh in d2 and d2[neigh] in one_hop1[id]:
                candidate_tri2.add(cur)
        elif cur[2] == e2:
            neigh = cur[0]
            if neigh in d2 and d2[neigh] in one_hop1[id]:
                candidate_tri2.add(cur)
        else:
            f_tri2.add(cur)
    
    r1, r2 = get_rfunc()
    
    for cur in tri1:
        if (cur[0] == id or cur[2] == id) and r1[cur[1]] > 0.7:
            candidate_tri1.add(cur)
        else:
            f_tri1.add(cur)
    for cur in tri2:
        if (cur[0] == d1[id] or cur[2] == d1[id]) and r2[cur[1]] > 0.7:
            candidate_tri2.add(cur)
        else:
            f_tri2.add(cur)
    '''
    nec_tri1 = defaultdict(set)
    # nec1 = set()
    # nec2 = set()
    nec_tri2 = defaultdict(set)
    
        
    print(len(candidate_tri1))
    print(len(nec1))
    candidate_tri1 -= nec1
    print(len(f_tri1))
    f_tri1 -= nec1
    print(len(candidate_tri1))
    candidate_tri2 -= nec2
    print(len(f_tri1))
    f_tri1 |= candidate_tri1
    print(len(f_tri1))
    f_tri2 |= candidate_tri2
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/nec_tri', 'w', encoding='utf-8') as f1:

        f1.write('-------------kg1----------------------\n')
        for tri1 in nec1:
            f1.write(ent[tri1[0]] + '\t' + r[tri1[1]] + '\t' + ent[tri1[2]] + '\n')
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2]+ '\n')
        f1.write('-------------kg2----------------------\n')
        for tri2 in nec2:
            f1.write(ent[tri2[0]] + '\t' + r[tri2[1]] + '\t' + ent[tri2[2]] + '\n')
            f1.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2]+ '\n')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/filter_tri1', 'w', encoding='utf-8') as f1, open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/filter_tri2', 'w', encoding='utf-8') as f2:
        for tri1 in f_tri1:
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2] + '\n')
        for tri2 in f_tri2:
            f2.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2] + '\n')


def tar_pair(id):
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    sim = torch.load('/data/xbtian/amie-master/zh-en/sim_cos.pt')
    r_rel = load_r_rel()
    one_hop1, neigh_r1, one_hop_r1 = get_1_hop(tri1)
    one_hop2, neigh_r2, one_hop_r2 = get_1_hop(tri2)
    r_align = defaultdict(set)
    r1 = defaultdict(set)
    r2 = defaultdict(set)
    
    f_tri1 = set()
    f_tri2 = set()
    
    
    candidate_tri1 = set()
    candidate_tri2 = set()
    can1 = defaultdict(set)
    can2 = defaultdict(set)
    e2 = d1[id]
    thred = 0.9
    for cur in tri1:
        if cur[0] == id or cur[2] == id:
            candidate_tri1.add(cur)
        else:
            f_tri1.add(cur)
    for cur in tri2:
        if cur[0] == e2 or cur[2] == e2:
            candidate_tri2.add(cur)
        else:
            f_tri2.add(cur)
    candidate_tri1_list = list(candidate_tri1)
    can1 = []
    for i in range(1, len(candidate_tri1_list) + 1):
        can1 += list((combinations(candidate_tri1_list, i)))
    candidate_tri2_list = list(candidate_tri2)
    can2 = []
    for i in range(1, len(candidate_tri2_list) + 1):
        can2 += list((combinations(candidate_tri2_list, i)))
    
    return candidate_tri1, candidate_tri2, can1, can2, f_tri1, f_tri2

def write_change(candidate_tri1, candidate_tri2, nec1, nec2, f_tri1, f_tri2):
    # print(len(candidate_tri1))
    # print(len(nec1))
    cur_tri1 = candidate_tri1 - nec1
    # print(len(f_tri1))
    # f_tri1 -= nec1
    # print(len(candidate_tri1))
    cur_tri2 = candidate_tri2 - nec2
    # print(candidate_tri2)
    # print(cur_tri2)
    # print(nec2)
    # print(len(f_tri1))
    tri1_f = f_tri1 | cur_tri1
    # print(len(f_tri2))
    tri2_f = f_tri2 | cur_tri2
    # print(len(f_tri1))
    # print(len(f_tri2))
    # print(len(nec1))
    # print(len(nec2))
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/nec_tri', 'w', encoding='utf-8') as f1:

        f1.write('-------------kg1----------------------\n')
        for tri1 in nec1:
            f1.write(ent[tri1[0]] + '\t' + r[tri1[1]] + '\t' + ent[tri1[2]] + '\n')
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2]+ '\n')
        f1.write('-------------kg2----------------------\n')
        for tri2 in nec2:
            f1.write(ent[tri2[0]] + '\t' + r[tri2[1]] + '\t' + ent[tri2[2]] + '\n')
            f1.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2]+ '\n')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/filter_tri1', 'w', encoding='utf-8') as f1, open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/filter_tri2', 'w', encoding='utf-8') as f2:
        for tri1 in tri1_f:
            f1.write(tri1[0] + '\t' + tri1[1] + '\t' + tri1[2] + '\n')
        for tri2 in tri2_f:
            f2.write(tri2[0] + '\t' + tri2[1] + '\t' + tri2[2] + '\n')
def co_1_to_2():
    from itertools import combinations
    file_in = 'zh-en/all_links'
    import math
    d1, d2 = read_link(file_in)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    tri_d1 = defaultdict(set)
    tri_d2 = defaultdict(set)
    for cur in tri1:
        tri_d1[cur[0]].add(cur[1])
    for cur in tri2:
        tri_d2[cur[0]].add(cur[1])
    co_rel = defaultdict(int)
    count_rel = defaultdict(int)
    for cur in d1:
        for cur_rel in tri_d1[cur]:
            count_rel[cur_rel] += 1
            for tar in list(combinations(list(tri_d2[d1[cur]]), 2)):
                co_rel[(cur_rel, tar)] += 1

    file_out = 'zh-en/co_rel_1_to_2'
    count =defaultdict(int)
    with open(file_out, 'w') as f:
        for cur in co_rel:
            f.write(cur[0] + '\t' + cur[1][0] + '\t' + cur[1][1] + '\t' + str(co_rel[cur])+'\n')
            count[co_rel[cur] / count_rel[cur[0]]] += 1
    
    x = []
    y = []
    for cur in count:
        # x.append(math.log2(cur))
        # y.append(math.log2(count[cur]))
        x.append(cur)
        y.append(count[cur])

    plt.scatter(x,y)
    plt.savefig('zh-en/rel_co_1_to_2.jpg')
    plt.show()

def relvance():
    neigh_sim = defaultdict(float)
    # l = read_list('zh-en/dual_amn')
    d1, d2 = read_link('zh-en/dual_amn')
    l = read_list('zh-en/test_links')
    p = read_pair('zh-en/test_links')
    golden = 0
    if golden:
        file_in = 'zh-en/all_links'
        file_out = 'zh-en/neigh_sim'
    else:
        file_in = 'zh-en/dual_amn'
        file_out = 'zh-en/neigh_sim_d'
    write_neigh_sim(file_in, file_out)
    with open(file_out, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for cur in lines:
            cur = cur.strip().split('\t')
            neigh_sim[(cur[0],cur[1])] = float(cur[2])
    gold_sim = defaultdict(float)
    top_sim = defaultdict(float)
    with open('/data/xbtian/ContEA-main/datasets/zh-en1/base/sim.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            cur = lines[i].strip().split('\t')
            gold_sim[(l[i][0],l[i][1])] = float(cur[3])
            top_sim[(l[i][0],l[int(cur[2])][1])] = float(cur[4])
    top_dict= {}
    gold_dict = {}
    neigh_dict = {}
    for cur in top_sim:
        top_dict[cur[0]] = ((cur[1], top_sim[cur]))
    for cur in gold_sim:
        gold_dict[cur[0]] = ((cur[1], gold_sim[cur]))
    for cur in neigh_sim:
        neigh_dict[cur[0]] = ((cur[1], neigh_sim[cur]))
    tmp_gold = defaultdict(list)
    tmp_top = defaultdict(list)
    y_top = []
    y_gold = []
    x = []
    diff = []
    n_d = {}
    for cur in top_dict:
        tmp_gold[int(neigh_dict[cur][1] * 100 / 10)].append(gold_dict[cur][1])
        tmp_top[int(neigh_dict[cur][1] * 100 / 10)].append(top_dict[cur][1])
    for cur in tmp_gold:
        x.append(cur)
        sum = 0
        for v in tmp_gold[cur]:
            sum += v
        y_gold.append(sum / len(tmp_gold[cur]))
    
    for cur in tmp_top:
        # x.append(cur)
        
        sum = 0
        for v in tmp_top[cur]:
            sum += v
        y_top.append(sum / len(tmp_top[cur]))
        n_d[cur] = sum / len(tmp_top[cur])
    # for cur in d1:
    new_pair = set()
    new_dict = {}
    for cur in top_dict:
        if top_dict[cur][1] >= n_d[int(neigh_dict[cur][1] * 100 / 10)] - 0.2 and int(neigh_dict[cur][1] * 100 / 10) > 1:
            new_pair.add((cur, top_dict[cur][0]))
            new_dict[cur] = top_dict[cur][0]
    print(len(new_pair & p) / len(p))
    print((len(new_pair) - len(new_pair & p)) / len(p))

    return new_pair, new_dict


    # plt.scatter(x,y_top, color = 'hotpink')
    # plt.scatter(x,y_gold,color = '#88c999')
    # plt.scatter(x,diff)
        
    
    '''
    x = []
    y = []
    for cur in top_sim:
        if cur in top_sim and cur not in neigh_sim:
            print(cur)
        if cur in top_sim and cur in neigh_sim:
            x.append(neigh_sim[cur])
            y.append(top_sim[cur])
        
    plt.scatter(x,y)
    '''
    plt.show()

def sample_right():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    pair1 = set()
    for cur in d1:
        pair1.add((cur, d1[cur]))
    file_in = 'zh-en/all_links'
    d1, d2 = read_link(file_in)
    pair2 = set()
    for cur in d1:
        pair2.add((cur, d1[cur]))
    p = len(pair1 & pair2) / len(pair1) 
    print('P : ', p)
    right = list(pair1 & pair2)
    deg1 = defaultdict(int)
    deg2 = defaultdict(int)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    for cur in tri1:
        deg1[cur[0]] += 1
        deg1[cur[2]] += 1
    for cur in tri2:
        deg2[cur[0]] += 1
        deg2[cur[2]] += 1
    sample = set()
    for cur in right:
        if deg1[cur[0]] > 10 and deg2[cur[1]] > 10:
            sample.add(cur)
    import random
    sample = random.sample(list(sample), 100)
    e1 = set()
    e2 = set()
    for cur in sample:
        e1.add(cur[0])
        e2.add(cur[1])
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    sample_tri1 = defaultdict(set)
    sample_tri2 = defaultdict(set)
    for cur in tri1:
        if cur[0] in e1:
            sample_tri1[cur[0]].add(cur)
        if cur[2] in e1:
            sample_tri1[cur[2]].add(cur)
    for cur in tri2:
        if cur[0] in e2:
            sample_tri2[cur[0]].add(cur)
        if cur[2] in e2:
            sample_tri2[cur[2]].add(cur)

    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair', 'w') as f:
        for cur in sample:
            f.write(cur[0] + '\t' + cur[1] + '\n')
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair_tri', 'w') as f:
        for cur in sample:
            f.write('-------------kg1----------------------\n')
            for t in sample_tri1[cur[0]]:
                f.write(ent[t[0]] + '\t' + r[t[1]] + '\t' + ent[t[2]] + '\n')
            f.write('-------------kg2----------------------\n')
            for t in sample_tri2[cur[1]]:
                f.write(ent[t[0]] + '\t' + r[t[1]] + '\t' + ent[t[2]] + '\n')

def get_result_align():
    file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    ent, _ = read_link('zh-en/ent_dict')
    with open('zh-en/res_dual', 'w') as f:
        for cur in d1:
            f.write(ent[cur] + '\t' + ent[d1[cur]] + '\n')


def get_2_kg(file):
    tri1 = set()
    tri2 = set()
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if int(cur[0]) < 19388:
                tri1.add((cur[0], cur[1], cur[2]))
            else:
                tri2.add((cur[0], cur[1], cur[2]))
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_triples_1', 'w') as f:
        for t in tri1:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_triples_2', 'w') as f:
        for t in tri2:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')


def get_2_no_nec_kg(file):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if int(cur[0]) < 19388:
                tri1 -= {(cur[0], cur[1], cur[2])}
            else:
                tri2 -= {(cur[0], cur[1], cur[2])}
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_triples_1', 'w') as f:
        for t in tri1:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_triples_2', 'w') as f:
        for t in tri2:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')

def get_2_nec_kg(file):
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    d1, d2 = read_link('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/sample_pair_v1')
    tar_tri1 = set()
    tar_tri2 = set()
    for cur in tri1:
        if cur[0] in d1 or cur[2] in d1:
            tar_tri1.add(cur)
    for cur in tri2:
        if cur[0] in d2 or cur[2] in d2:
            tar_tri2.add(cur)
    tri1 -= tar_tri1
    tri2 -= tar_tri2
    count1 = 0
    count2 = 0
    nec_tri1 = set()
    nec_tri2 = set()
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split('\t')
            if int(cur[0]) < 19388:
                tri1.add((cur[0], cur[1], cur[2]))
                nec_tri1.add((cur[0], cur[1], cur[2]))
            else:
                tri2.add((cur[0], cur[1], cur[2]))
                nec_tri2.add((cur[0], cur[1], cur[2]))
                count2 += 1
    print('sparsity :', (len(nec_tri1)) / (len(tar_tri1)), (len(nec_tri2)) / (len(tar_tri2)))
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_triples_1_nec', 'w') as f:
        for t in tri1:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')
    with open('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_triples_2_nec', 'w') as f:
        for t in tri2:
            f.write(t[0] + '\t' + t[1] + '\t' + t[2] + '\n')


def sample_align(file_in, file_out):
    # file_in = 'zh-en/dual_amn'
    d1, d2 = read_link(file_in)
    pair1 = set()
    for cur in d1:
        pair1.add((cur, d1[cur]))
    file_in = 'zh-en/all_links'
    d1, d2 = read_link(file_in)
    pair2 = set()
    for cur in d1:
        pair2.add((cur, d1[cur]))
    p = len(pair1 & pair2) / len(pair1) 
    print('P : ', p)
    right = list(pair1 & pair2)
    deg1 = defaultdict(int)
    deg2 = defaultdict(int)
    tri1 = read_tri('zh-en/triples_1')
    tri2 = read_tri('zh-en/triples_2')
    for cur in tri1:
        deg1[cur[0]] += 1
        deg1[cur[2]] += 1
    for cur in tri2:
        deg2[cur[0]] += 1
        deg2[cur[2]] += 1
    sample = set()
    for cur in right:
        if deg1[cur[0]] > 10 and deg2[cur[1]] > 10:
            sample.add(cur)
    import random
    sample = random.sample(list(sample), 100)
    e1 = set()
    e2 = set()
    for cur in sample:
        e1.add(cur[0])
        e2.add(cur[1])
    r, _ = read_link('zh-en/rel_dict')
    ent, _ = read_link('zh-en/ent_dict')
    sample_tri1 = defaultdict(set)
    sample_tri2 = defaultdict(set)
    for cur in tri1:
        if cur[0] in e1:
            sample_tri1[cur[0]].add(cur)
        if cur[2] in e1:
            sample_tri1[cur[2]].add(cur)
    for cur in tri2:
        if cur[0] in e2:
            sample_tri2[cur[0]].add(cur)
        if cur[2] in e2:
            sample_tri2[cur[2]].add(cur)

    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair', 'w') as f:
        for cur in sample:
            f.write(cur[0] + '\t' + cur[1] + '\n')
    with open('/data/xbtian/ContEA-main/datasets/zh-en_f/base/sample_pair_tri', 'w') as f:
        for cur in sample:
            f.write('-------------kg1----------------------\n')
            for t in sample_tri1[cur[0]]:
                f.write(ent[t[0]] + '\t' + r[t[1]] + '\t' + ent[t[2]] + '\n')
            f.write('-------------kg2----------------------\n')
            for t in sample_tri2[cur[1]]:
                f.write(ent[t[0]] + '\t' + r[t[1]] + '\t' + ent[t[2]] + '\n')

def get_name(file, file_out):
    ent_dict, _ = read_link('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/ent_dict')
    pair_dict, _ = read_link('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/ori_pair.txt')
    with open(file_out, 'w') as f:
        for cur in pair_dict:
            f.write(ent_dict[cur] + '\t' + ent_dict[pair_dict[cur]] + '\n')

def rel_neigh(file1, file2):
    p1 = read_pair(file1)
    p2 = read_pair(file2)
    d1, d2 = read_link(file1)
    d3, d4 = read_link(file2)
    d5, d6 = read_link('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_links')
    align_set = defaultdict(set)
    for cur in p2:
        if cur[0] in d1:
            align_set[cur[0]].add(cur[1])
    count2 = 0
    for cur in d1:
        count1 = 0
        
        # print('global align : ', cur + '\t' + d1[cur])
        if d1[cur] != d5[cur]:
            count1 = 1
        # print('local align :')
        for local in align_set[cur]:
            if local == d5[cur] and count1 == 1:
                print(cur + '\t' +local)
                count2 += 1
    print(count2)

# rel_neigh('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/dualamn_pair.txt', '/data/xbtian/ContEA-explain/datasets/zh-en_f/base/match_neighbors0')
# get_name('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/ori_pair.txt', '/data/xbtian/ContEA-explain/datasets/zh-en_f/base/dualamn_pair.txt')
# get_2_nec_kg('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/nec_tri')
# get_2_no_nec_kg('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/nec_tri')
# get_2_kg('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_kgs_no')
# get_2_kg('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/test_kgs')
# get_result_align()
# sample_right()
# baseline_tri()
# baseline_tri_align()
# baseline_tri_2_hop()
# baseline_tri_sim()
# count_rel()
# change_tar('36')
# count_rel()
# sample_right()
# co_1_to_2()
# relvance()
# explain_neigh_sim()
# get_entity_tri('http://zh.dbpedia.org/resource/張力尹', 0)
# get_entity_tri('http://dbpedia.org/resource/Puplinge', 1)
# r1_func = get_r_func('zh-en/r_func1_id')
# r2_func = get_r_func('zh-en/r_func2_id')
# write_neigh_sim()
# print('explain 2')
# get_explain_1(r1_func, r2_func)
# get_explain_all(r1_func, r2_func)
# get_explain_count()
# get_confidence_filter()
# r_align_0 = get_r_conf_all('zh-en/r_con_id', 0)
# r1_func = get_r_func('zh-en/r_func1_id')
# get_confidence_1_filter(r1_func)
# r2_func = get_r_func('zh-en/r_func2_id')
# get_confidence_3_filter(r1_func, r2_func)
# get_confidence_2_filter(r2_func)
# neigh_sim()
# comp_sim()
# explain_neigh_sim()











#get_confidence_3()
# get_confidence_2()
# get_explain()
# get_confidence()
# get_r_func()
# get_right()
# get_explain_all()
