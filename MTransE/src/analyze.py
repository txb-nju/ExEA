from count import read_link, read_tri
from preprocessing1 import DBpDataset
from collections import defaultdict
def rest_ent(file):
    cur_link, cur_link_r = read_link('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/pair_zh.txt')
    G_dataset = DBpDataset('/data/xbtian/Explain/MTransE/datasets/OpenEA/EN_DE_15K', device='cuda', pair='/pair', lang='zh')
    ent1 = set()

    for cur in cur_link:
        ent1.add(cur)
        
    rest_ent = set()
    for cur in G_dataset.test_link:
        if cur not in ent1:
            rest_ent.add(cur)
    for cur in rest_ent:
        for tri in G_dataset.gid[int(cur)]:
            G_dataset.read_triple_name(tri)
        print('-------------------------')
        for tri in G_dataset.gid[int(G_dataset.test_link[cur])]:
            G_dataset.read_triple_name(tri)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        if G_dataset.test_link[cur] in cur_link_r:
            for tri in G_dataset.gid[int(cur_link_r[G_dataset.test_link[cur]])]:
                G_dataset.read_triple_name(tri)
        print('************************')

def triangle_count():
    G_dataset = DBpDataset('/data/xbtian/Explain/MTransE/datasets/dbp_z_e', device='cuda', pair='/pair', lang='zh')
    r = defaultdict(set)
    for cur in G_dataset.kg1:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    for cur in G_dataset.kg2:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    conflict = set()
    conflict_id = set()
    co_r = set()
    for cur in G_dataset.ent_dict:
        cur_list = G_dataset.gid[int(cur)]
        neigh = set()
        for i in range(len(cur_list)):
            if cur_list[i][0] == int(cur):
                neigh.add(cur_list[i][2])
            else:
                neigh.add(cur_list[i][0])
        neigh = list(neigh)
        cur_r = set()
        for i in range(len(neigh)):
            e1 = neigh[i]
            cur_r = r[(int(cur), e1)]
            cur_r = list(cur_r)
            for i in range(len(cur_r) - 1):
                for j in range(i + 1, len(cur_r)):
                    co_r.add((cur_r[i], cur_r[j]))
                    co_r.add((cur_r[j], cur_r[i]))
        for i in range(len(neigh) - 1):
            e1 = neigh[i]
            for j in range(i + 1, len(neigh)):
                e2 = neigh[j]
                if len(r[(e1, e2)]) > 0:
                    for cur_r1 in r[(e1, int(cur))]:
                        for cur_r2 in r[(e2, int(cur))]:
                            if cur_r1 != cur_r2:
                                conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                                conflict_id.add((cur_r1, cur_r2))
    # print(len(co_r))
    conflict_id = conflict_id.difference(co_r)
    for cur in conflict:
        # print(str(cur[0]) ,str(cur[1]))
        print(str(cur[0]) + '\t' + str(cur[1]))
        

def line_count():
    G_dataset = DBpDataset('/data/xbtian/Explain/MTransE/datasets/dbp_z_e', device='cuda', pair='/pair', lang='zh')
    r = defaultdict(set)
    for cur in G_dataset.kg1:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    for cur in G_dataset.kg2:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    conflict = set()
    conflict_id = set()
    co_r = set()
    for cur in G_dataset.ent_dict:
        cur_list = G_dataset.gid[int(cur)]
        neigh = set()
        for i in range(len(cur_list)):
            if cur_list[i][0] == int(cur):
                neigh.add(cur_list[i][2])
            else:
                neigh.add(cur_list[i][0])
        neigh = list(neigh)
        cur_r = set()
        for i in range(len(neigh)):
            e1 = neigh[i]
            cur_r_l = r[(int(cur), e1)]
            cur_r_l = list(cur_r_l)
            for i in range(len(cur_r_l) - 1):
                for j in range(i + 1, len(cur_r_l)):
                    co_r.add((cur_r_l[i], 0, cur_r_l[j], 0))
                    co_r.add((cur_r_l[j], 0, cur_r_l[i], 0))
            cur_r_r = r[(e1, int(cur))]
            cur_r_r = list(cur_r_r)
            for i in range(len(cur_r_r) - 1):
                for j in range(i + 1, len(cur_r_r)):
                    co_r.add((cur_r_r[i], 1, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_r[i], 1))
            
            for i in range(len(cur_r_l)):
                for j in range(len(cur_r_r)):
                    co_r.add((cur_r_l[i], 0, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_l[i], 0))
                    # co_r.add((cur_r[j], cur_r[i]))
        for i in range(len(neigh) - 1):
            e1 = neigh[i]
            for j in range(i + 1, len(neigh)):
                e2 = neigh[j]
                if len(r[(e1, e2)]) > 0:
                    for cur_r1 in r[(e1, int(cur))]:
                        for cur_r2 in r[(e2, int(cur))]:
                            # if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 0, cur_r2, 0))
                            conflict_id.add((cur_r2, 0, cur_r1, 0))
                    for cur_r1 in r[(int(cur), e1)]:
                        for cur_r2 in r[(e2, int(cur))]:
                            # if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 1, cur_r2, 0))
                            conflict_id.add((cur_r2, 0, cur_r1, 1))
                    for cur_r1 in r[(e1, int(cur))]:
                        for cur_r2 in r[(int(cur), e2)]:
                            # if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 0, cur_r2, 1))
                            conflict_id.add((cur_r2, 1, cur_r1, 0))
                    for cur_r1 in r[(int(cur), e1)]:
                        for cur_r2 in r[(int(cur), e2)]:
                            # if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 1, cur_r2, 1))
                            conflict_id.add((cur_r2, 1, cur_r1, 1))
    # print(len(co_r))
    print(len(co_r), len(conflict_id))
    line = co_r.difference(conflict_id)
    print(len(line))
    for cur in line:
        # print(str(cur[0]) ,str(cur[1]))
        print(G_dataset.r_dict[cur[0]]+ '\t' + str(cur[1]) + '\t' + G_dataset.r_dict[cur[2]] + '\t' + str(cur[3]))
        # print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))

def triangle_count1():
    G_dataset = DBpDataset('/data/xbtian/Explain/MTransE/datasets/dbp_f_e', device='cuda', pair='/pair', lang='fr')
    r = defaultdict(set)
    for cur in G_dataset.kg1:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    for cur in G_dataset.kg2:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    conflict = set()
    conflict_id = set()
    co_r = set()
    for cur in G_dataset.ent_dict:
        cur_list = G_dataset.gid[int(cur)]
        neigh = set()
        for i in range(len(cur_list)):
            if cur_list[i][0] == int(cur):
                neigh.add(cur_list[i][2])
            else:
                neigh.add(cur_list[i][0])
        neigh = list(neigh)
        cur_r = set()
        for i in range(len(neigh)):
            e1 = neigh[i]
            cur_r_l = r[(int(cur), e1)]
            cur_r_l = list(cur_r_l)
            for i in range(len(cur_r_l) - 1):
                for j in range(i + 1, len(cur_r_l)):
                    co_r.add((cur_r_l[i], 0, cur_r_l[j], 0))
                    co_r.add((cur_r_l[j], 0, cur_r_l[i], 0))
            cur_r_r = r[(e1, int(cur))]
            cur_r_r = list(cur_r_r)
            for i in range(len(cur_r_r) - 1):
                for j in range(i + 1, len(cur_r_r)):
                    co_r.add((cur_r_r[i], 1, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_r[i], 1))
            
            for i in range(len(cur_r_l)):
                for j in range(len(cur_r_r)):
                    co_r.add((cur_r_l[i], 0, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_l[i], 0))
                    # co_r.add((cur_r[j], cur_r[i]))
        for i in range(len(neigh) - 1):
            e1 = neigh[i]
            for j in range(i + 1, len(neigh)):
                e2 = neigh[j]
                if len(r[(e1, e2)]) > 0:
                    for cur_r1 in r[(e1, int(cur))]:
                        for cur_r2 in r[(e2, int(cur))]:
                            if cur_r1 != cur_r2:
                            # if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                                conflict_id.add((cur_r1, 0, cur_r2, 0))
                                conflict_id.add((cur_r2, 0, cur_r1, 0))
                    for cur_r1 in r[(int(cur), e1)]:
                        for cur_r2 in r[(e2, int(cur))]:
                            if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                                conflict_id.add((cur_r1, 1, cur_r2, 0))
                                conflict_id.add((cur_r2, 0, cur_r1, 1))
                    for cur_r1 in r[(e1, int(cur))]:
                        for cur_r2 in r[(int(cur), e2)]:
                            if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                                conflict_id.add((cur_r1, 0, cur_r2, 1))
                                conflict_id.add((cur_r2, 1, cur_r1, 0))
                    for cur_r1 in r[(int(cur), e1)]:
                        for cur_r2 in r[(int(cur), e2)]:
                            if cur_r1 != cur_r2:
                            # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                                conflict_id.add((cur_r1, 1, cur_r2, 1))
                                conflict_id.add((cur_r2, 1, cur_r1, 1))
    # print(len(co_r))
    # print(len(co_r), len(conflict_id))
    traingle = conflict_id.difference(co_r)
    # print(len(line))
    for cur in traingle:
        # print(str(cur[0]) ,str(cur[1]))
        # print(G_dataset.r_dict[cur[0]]+ '\t' + str(cur[1]) + '\t' + G_dataset.r_dict[cur[2]] + '\t' + str(cur[3]))
        print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
    
def triangle_count2():
    G_dataset = DBpDataset('/data/xbtian/Explain/MTransE/datasets/dbp_z_e', device='cuda', pair='/pair', lang='zh')
    r = defaultdict(set)
    for cur in G_dataset.kg1:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    for cur in G_dataset.kg2:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    conflict = set()
    conflict_id = set()
    co_r = set()

    for cur in G_dataset.ent_dict:
        cur_list = G_dataset.gid[int(cur)]
        neigh = set()
        for i in range(len(cur_list)):
            if cur_list[i][0] == int(cur):
                neigh.add(cur_list[i][2])
            else:
                neigh.add(cur_list[i][0])
        neigh = list(neigh)
        cur_r = set()
        for i in range(len(neigh)):
            e1 = neigh[i]
            cur_r_l = r[(int(cur), e1)]
            cur_r_l = list(cur_r_l)
            for i in range(len(cur_r_l) - 1):
                for j in range(i + 1, len(cur_r_l)):
                    co_r.add((cur_r_l[i], 0, cur_r_l[j], 0))
                    co_r.add((cur_r_l[j], 0, cur_r_l[i], 0))
            cur_r_r = r[(e1, int(cur))]
            cur_r_r = list(cur_r_r)
            for i in range(len(cur_r_r) - 1):
                for j in range(i + 1, len(cur_r_r)):
                    co_r.add((cur_r_r[i], 1, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_r[i], 1))
            
            for i in range(len(cur_r_l)):
                for j in range(len(cur_r_r)):
                    co_r.add((cur_r_l[i], 0, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_l[i], 0))
                    # co_r.add((cur_r[j], cur_r[i]))
    r_dict1, _ = G_dataset.read_dict('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/rel_dict1')
    r_dict2, _ = G_dataset.read_dict('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/rel_dict2')
    r_set1 = set()
    r_set2 = set()
    for cur in r_dict1:
        r_set1.add(cur)
    for cur in r_dict2:
        r_set2.add(cur)
    r_set1 = list(r_set1)
    r_set2 = list(r_set2)
    for i in range(len(r_set1) - 1):
        for j in range(i + 1, len(r_set1)):
            conflict_id.add((r_set1[i], 0, r_set1[j], 0))
            conflict_id.add((r_set1[i], 1, r_set1[j], 0))
            conflict_id.add((r_set1[i], 0, r_set1[j], 1))
            conflict_id.add((r_set1[i], 1, r_set1[j], 1))
            conflict_id.add((r_set1[j], 0, r_set1[i], 0))
            conflict_id.add((r_set1[j], 1, r_set1[i], 0))
            conflict_id.add((r_set1[j], 0, r_set1[i], 1))
            conflict_id.add((r_set1[j], 1, r_set1[i], 1))
    for i in range(len(r_set2) - 1):
        for j in range(i + 1, len(r_set2)):
            conflict_id.add((r_set2[i], 0, r_set2[j], 0))
            conflict_id.add((r_set2[i], 1, r_set2[j], 0))
            conflict_id.add((r_set2[i], 0, r_set2[j], 1))
            conflict_id.add((r_set2[i], 1, r_set2[j], 1))
            conflict_id.add((r_set2[j], 0, r_set2[i], 0))
            conflict_id.add((r_set2[j], 1, r_set2[i], 0))
            conflict_id.add((r_set2[j], 0, r_set2[i], 1))
            conflict_id.add((r_set2[j], 1, r_set2[i], 1))
            
    # print(len(co_r))
    # print(len(co_r), len(conflict_id))
    traingle = conflict_id.difference(co_r)
    # print(len(line))
    for cur in traingle:
        # print(str(cur[0]) ,str(cur[1]))
        # print(G_dataset.r_dict[cur[0]]+ '\t' + str(cur[1]) + '\t' + G_dataset.r_dict[cur[2]] + '\t' + str(cur[3]))
        print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
# rest_ent('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/pair.txt')
# triangle_count()
def triangle_count3():
    G_dataset = DBpDataset('../datasets/z_e_2', device='cuda', pair='/pair', lang='zh2')
    r = defaultdict(set)
    for cur in G_dataset.kg1:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    for cur in G_dataset.kg2:
        r[(int(cur[0]), int(cur[2]))].add(int(cur[1]))
        # r[(int(cur[2]), int(cur[0]))].add(int(cur[1]))
    conflict = set()
    conflict_id = set()
    co_r = set()
    for cur in G_dataset.ent_dict:
        cur_list = G_dataset.gid[int(cur)]
        neigh = set()
        for i in range(len(cur_list)):
            if cur_list[i][0] == int(cur):
                neigh.add(cur_list[i][2])
            else:
                neigh.add(cur_list[i][0])
        neigh = list(neigh)
        cur_r = set()
        for i in range(len(neigh)):
            e1 = neigh[i]
            cur_r_l = r[(int(cur), e1)]
            cur_r_l = list(cur_r_l)
            for i in range(len(cur_r_l) - 1):
                for j in range(i + 1, len(cur_r_l)):
                    co_r.add((cur_r_l[i], 0, cur_r_l[j], 0))
                    co_r.add((cur_r_l[j], 0, cur_r_l[i], 0))
            cur_r_r = r[(e1, int(cur))]
            cur_r_r = list(cur_r_r)
            for i in range(len(cur_r_r) - 1):
                for j in range(i + 1, len(cur_r_r)):
                    co_r.add((cur_r_r[i], 1, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_r[i], 1))
            
            for i in range(len(cur_r_l)):
                for j in range(len(cur_r_r)):
                    co_r.add((cur_r_l[i], 0, cur_r_r[j], 1))
                    co_r.add((cur_r_r[j], 1, cur_r_l[i], 0))
                    # co_r.add((cur_r[j], cur_r[i]))
        for i in range(len(neigh) - 1):
            e1 = neigh[i]
            for j in range(i + 1, len(neigh)):
                e2 = neigh[j]
                for cur_r1 in r[(e1, int(cur))]:
                    for cur_r2 in r[(e2, int(cur))]:
                        if cur_r1 != cur_r2:
                        # if cur_r1 != cur_r2:
                        # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 0, cur_r2, 0))
                            conflict_id.add((cur_r2, 0, cur_r1, 0))
                for cur_r1 in r[(int(cur), e1)]:
                    for cur_r2 in r[(e2, int(cur))]:
                        if cur_r1 != cur_r2:
                        # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 1, cur_r2, 0))
                            conflict_id.add((cur_r2, 0, cur_r1, 1))
                for cur_r1 in r[(e1, int(cur))]:
                    for cur_r2 in r[(int(cur), e2)]:
                        if cur_r1 != cur_r2:
                        # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 0, cur_r2, 1))
                            conflict_id.add((cur_r2, 1, cur_r1, 0))
                for cur_r1 in r[(int(cur), e1)]:
                    for cur_r2 in r[(int(cur), e2)]:
                        if cur_r1 != cur_r2:
                        # conflict.add((G_dataset.r_dict[cur_r1], G_dataset.r_dict[cur_r2]))
                            conflict_id.add((cur_r1, 1, cur_r2, 1))
                            conflict_id.add((cur_r2, 1, cur_r1, 1))
    # print(len(co_r))
    # print(len(co_r), len(conflict_id))
    traingle = conflict_id.difference(co_r)
    # print(len(line))
    for cur in traingle:
        # print(str(cur[0]) ,str(cur[1]))
        print(G_dataset.r_dict[cur[0]]+ '\t' + str(cur[1]) + '\t' + G_dataset.r_dict[cur[2]] + '\t' + str(cur[3]))
        # print(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\t' + str(cur[3]))
triangle_count3()