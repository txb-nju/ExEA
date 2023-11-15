import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import math
import tqdm
import sys
# from model import Encoder_Model
import warnings
import argparse
import time
from collections import defaultdict
import anchor
from anchor import anchor_tabular
from preprocessing import DBpDataset
from count import read_tri, read_link
import numpy as np
import logging
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
import time
from tqdm import trange
import copy
from count import read_list
import matplotlib.pyplot as plt
from itertools import combinations
import random
import numpy as np
import argparse
import difflib
import pandas as pd
from LORE import util
from LORE.prepare_dataset import prepare_EA_dataset
from LORE import lore
from LORE.neighbor_generator import genetic_neighborhood


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Proxy:
    def __init__(self, embed1, embed2):
        self.embed1 = embed1
        self.embed2 = embed2


    def aggregation(self, tri1, tri2):
        return self.embed1[tri1].mean(dim = 0), self.embed2[tri2].mean(dim = 0)
    
    def all_aggregation(self):
        return self.embed1.mean(dim = 0), self.embed2.mean(dim = 0)
    
    def sim(self, tri1, tri2):
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        return F.cosine_similarity(self.embed1[tri1].mean(dim = 0), self.embed2[tri2].mean(dim = 0), dim=0)

    def predict(self, x, split):
        tri1 = []
        tri2 = []
        for i in range(len(x)):
            if x[i] == 1:
                tri1.append(player)
                tri2.append(player - split)
        
        sim = self.sim_kernel(tri1, tri2)
    def mask_sim(self, mask, split):

        return F.cosine_similarity((self.embed1 * mask[:split].unsqueeze(1)).mean(dim = 0), (self.embed2 * mask[split:].unsqueeze(1)).mean(dim = 0), dim=0)



class Shapley_Value:
    def __init__(self, model, num_players, players, split):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split

    def MTC(self, num_simulations):
        shapley_values = np.zeros(self.num_players)
        for _ in range(num_simulations):
        # 生成随机排列的玩家列表
            players = np.random.permutation(self.num_players)
            # print(players, split)
            # 初始化联盟价值和玩家计数器
            coalition_value = 0
            player_count = 0
            tri1 = []
            tri2 = []
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)

                coalition_value_with_player = self.model.sim(tri1, tri2)
                # print(tri1, tri2)
                # print(coalition_value_with_player)
                marginal_contribution = coalition_value_with_player - coalition_value

                # 计算当前玩家的 Shapley 值
                shapley_values[player] += marginal_contribution / num_simulations

                # 更新联盟价值和玩家计数器
                coalition_value = coalition_value_with_player
                # player_count += 1
        return shapley_values

class LIME:
    def __init__(self, model, num_players, players, split, e1, e2, embed):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.e1 = e1
        self.e2 = e2
        self.embed = embed

    def sim_kernel(self, tri1, tri2):
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        tmp_e1, tmp_e2 = self.model.aggregation(tri1, tri2)
        
        sim1 = F.cosine_similarity(tmp_e1, self.embed[self.e1], dim=0)
        sim2 = F.cosine_similarity(tmp_e2, self.embed[self.e2], dim=0)
        print(tri1, tri2, sim1, sim2)
        return (sim1 + sim2) / 2

    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)
            
            sim = self.sim_kernel(tri1, tri2)
            if sim == 0:
                continue
            mask.append(cur_mask)
            Y.append(self.model.sim(tri1, tri2))
            
            pi.append(float(sim))
                    
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        I = torch.eye(Z.shape[1])
        # print(pi)
        # print(Z,Y,I)
        # exit(0)
        # pi = torch.Tensor(pi)
        # pi = torch.diag(pi)
        # print(np.array(pi) + 1)
        # print(Z)
        # print(Y)
        reg = LinearRegression().fit(Z,Y,np.array(pi) + 1)
        # print(reg.coef_)
        # exit(0)
        # pi = I
        res = reg.coef_
        # res = torch.mm(torch.inverse(torch.mm(torch.mm(Z.t(),pi),Z) + I), torch.mm(torch.mm(Z.t(),pi), Y.unsqueeze(1)))
        # res = torch.mm(torch.inverse(torch.mm(Z.t(),Z) + I), torch.mm(Z.t(), Y.unsqueeze(1)))
        return res

class LLM:
    def __init__(self, model, num_players, players, split, e1, e2, embed):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.e1 = e1
        self.e2 = e2
        self.embed = embed

    def sim_kernel(self, tri1, tri2):
        if len(tri1) == 0 or len(tri2) == 0:
            return 0
        tmp_e1, tmp_e2 = self.model.aggregation(tri1, tri2)
        
        sim1 = F.cosine_similarity(tmp_e1, self.embed[self.e1], dim=0)
        sim2 = F.cosine_similarity(tmp_e2, self.embed[self.e2], dim=0)
        # print(tri1, tri2, sim1, sim2)
        return (sim1 + sim2) / 2

    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)
            
            sim = self.sim_kernel(tri1, tri2)
            if sim == 0:
                continue
            mask.append(cur_mask)
            Y.append(self.model.sim(tri1, tri2))
            
            pi.append(float(sim))
                    
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        return Z, Y

class Anchor:
    def __init__(self, model, num_players, players, split, e1, e2, embed):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.e1 = e1
        self.e2 = e2
        self.embed = embed

    def compute(self, sample_nums, alpha):
        mask = []
        Y = []
        pi = []
        pred = {}
        def encoder_fn(X):
            tri = []
            # print(X)
            for x in X:
                tri1 = []
                tri2 = []
                for i in range(len(x)):
                    if x[i] == 1:
                        if i < self.split:
                            tri1.append(i)
                        else:
                            tri2.append(i - self.split)
                tri.append((tri1, tri2))
            return tri
        def predict(tri):
            res = []
            # print(tri)
            for tri1, tri2 in tri:
                # print(tri1, tri2)
                sim = self.model.sim(tri1, tri2)
                if sim >= alpha:
                    c = 1
                else:
                    c = 0
                res.append(c)
            return np.array(res)
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)
            
            sim = self.model.sim(tri1, tri2)
            if sim >= alpha:
                c = 1
            else:
                c = 0
            pred[tuple(cur_mask)] = c
            
            mask.append(cur_mask)
            Y.append(c)
                    
        Z = np.array(mask)
        self.explainer = anchor_tabular.AnchorTabularExplainer(
            class_names=[0,1],
            feature_names=list(range(self.num_players)),
            train_data=Z,
            categorical_names={},
            encoder_fn=encoder_fn
        )
        x = np.array([[1] * (self.num_players)])
        
        explanation = self.explainer.explain_instance(x, predict)

        # 打印Anchor规则
        # print(explanation.names())
        return explanation.names()

class BLACK:
    def __init__(self, model, split, alpha):
        self.model = model
        self.split = split
        self.alpha = alpha
    
    def predict(self, X):
        tri = []
        # print(X)
        for x in X:
            tri1 = []
            tri2 = []
            for i in range(len(x)):
                if x[i] == 1:
                    if i < self.split:
                        tri1.append(i)
                    else:
                        tri2.append(i - self.split)
            tri.append((tri1, tri2))
        res = []
        # print(tri)
        for tri1, tri2 in tri:
            # print(tri1, tri2)
            sim = self.model.sim(tri1, tri2)
            if sim >= self.alpha:
                c = 1
            else:
                c = 0
            res.append(c)
        return np.array(res)
        
class LORE:
    def __init__(self, model, num_players, players, split, e1, e2, embed, lang):
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.e1 = e1
        self.e2 = e2
        self.embed = embed
        self.lang = lang

    def compute(self, sample_nums, alpha):
        data = []
        Y = []
        pi = []
        pred = []
        
        data.append([1] * (self.num_players + 1))
        X2E = []
        X2E.append([1] * (self.num_players))
        for _ in range(sample_nums):
            # players = np.random.permutation(random.randint(0, self.num_players))
            players = np.random.choice(self.num_players,random.randint(0, self.num_players), replace=False)

            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players + 1)
            cur_x = [0] * (self.num_players)
            for player in players:
                cur_mask[player] = 1
                cur_x[player] = 1
                if player < self.split:
                    tri1.append(player)
                else:
                    tri2.append(player - self.split)
            
            sim = self.model.sim(tri1, tri2)
            if sim >= alpha:
                c = 1
            else:
                c = 0
            Y.append(sim)
            pred.append(c)
            cur_mask[self.num_players] = c
            data.append(cur_mask)
            X2E.append(cur_x)
        X2E = np.array(X2E)
        # print(pred)
        if max(pred) == 0 or min(pred) == 1:
            alpha = sum(Y) / len(Y)
            for j in range(len(Y)):
                if Y[j] > alpha:
                    data[j + 1][-1] = 1
                else:
                    data[j + 1][-1] = 0
        # print(data)
        feature = [str(i) for i in range(self.num_players)]
        feature.append('class')
        df = pd.DataFrame(data, columns=feature)
        dataset = prepare_EA_dataset(df, self.lang)
        blackbox = BLACK(self.model, self.split, alpha)
        explanation, infos = lore.explain(0, X2E, dataset, blackbox,
                                      ng_function=genetic_neighborhood,
                                      discrete_use_probabilities=True,
                                      continuous_function_estimation=False,
                                      returns_infos=True,
                                      path='', sep=';', log=False)
        if explanation == None:
            return None, None
        print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
        return explanation[0][1], explanation[0][0]['class']
class KernelSHAP:
    def __init__(self, model, e1, e2, num_players, players, split, p, e_dict, r_dict, dataset, embed):
        self.model = model
        self.e1 = e1
        self.e2 = e2
        self.num_players = num_players
        self.players = players
        self.split = split
        self.p = p
        self.e_dict = e_dict
        self.r_dict = r_dict
        self.dataset = dataset
        self.embed = embed

    def value(self, tri1, tri2):
        try:
            ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.dataset.reconstruct_search(self.e_dict[self.e1], self.e_dict[self.e2], tri1, tri2, True, len(self.e_dict), len(self.r_dict))
            # print(node_size,adj_list, model.ent_embedding.weight.shape )
            proxy_e1, proxy_e2 = self.model.get_embeddings([self.e_dict[self.e1]], [self.e_dict[self.e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
            v_suf = F.cosine_similarity(proxy_e1[0], proxy_e2[0], dim=0)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            if "浮点数" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception
            v_suf = -2

        return v_suf
    def sim_kernel(self, tri1, tri2):
        z = len(tri1) + len(tri2)
        # print(z, self.num_players)
        sim = (self.num_players - 1) / (comb(self.num_players, z) * z * (self.num_players - z))

        return sim
    def compute(self, sample_nums):
        mask = []
        Y = []
        pi = []
        for _ in range(sample_nums):
            players = np.random.permutation(random.randint(0, self.num_players))
            if len(players) == self.num_players or len(players) == 0:
                continue
            tri1 = []
            tri2 = []
            cur_mask = [0] * (self.num_players)
            for player in players:
                # 计算当前联盟中添加玩家后的价值差异
                cur_mask[player] = 1
                if player < self.split:
                    tri1.append(self.p[player])
                else:
                    tri2.append(self.p[player])
            # print('compute')
            # print(tri1, tri2)
            v = self.value(tri1, tri2)
            # print(v)
            sim = self.sim_kernel(tri1, tri2)
            # print(sim)
            
            if v == -2:
                continue
            pi.append(sim)
            mask.append(cur_mask)
            Y.append(v)
                    
        # print('end')
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        I = torch.eye(Z.shape[1])
        pi = torch.Tensor(pi)
        pi = torch.diag(pi)
        # print('cur')
        res = torch.mm(torch.inverse(torch.mm(torch.mm(Z.t(),pi),Z) + I), torch.mm(torch.mm(Z.t(),pi), Y.unsqueeze(1)))
        return res
            
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class EAExplainer(torch.nn.Module):
    def __init__(self, model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, split, splitr=0, lang='zh'):
        super(EAExplainer, self).__init__()
        self.model_name = model_name
        self.dist = nn.PairwiseDistance(p=2)
        self.split = split
        self.splitr = splitr
        self.lang = lang
        self.conflict_r_pair = G_dataset.conflict_r_pair
        self.conflict_id = G_dataset.conflict_id
        self.line_id = G_dataset.line_id
        if lang == 'zh':
            loadData = np.load('../saved_model/ent_embeds.npy')
            r_embed = np.load('../saved_model/rel_embeds.npy')
            mapping = np.load('../saved_model/mapping_mat.npy')
        elif lang == 'ja':
            loadData = np.load('../saved_model/ent_embeds_ja.npy')
            r_embed = np.load('../saved_model/rel_embeds_ja.npy')
            mapping = np.load('../saved_model/mapping_mat_ja.npy')
        elif lang == 'fr':
            loadData = np.load('../saved_model/ent_embeds_fr.npy')
            r_embed = np.load('../saved_model/rel_embeds_fr.npy')
            mapping = np.load('../saved_model/mapping_mat_fr.npy')
        elif lang == 'de':
            loadData = np.load('../saved_model/ent_embeds_de.npy')
            r_embed = np.load('../saved_model/rel_embeds_de.npy')
            mapping = np.load('../saved_model/mapping_mat_de.npy')
        elif lang == 'y':
            loadData = np.load('../saved_model/ent_embeds_y.npy')
            r_embed = np.load('../saved_model/rel_embeds_y.npy')
            mapping = np.load('../saved_model/mapping_mat_y.npy')
        elif lang == 'w':
            loadData = np.load('../saved_model/ent_embeds_w.npy')
            r_embed = np.load('../saved_model/rel_embeds_w.npy')
            mapping = np.load('../saved_model/mapping_mat_w.npy')
        elif lang == 'w2':
            loadData = np.load('../saved_model/ent_embeds_w2.npy')
            r_embed = np.load('../saved_model/rel_embeds_w2.npy')
            mapping = np.load('../saved_model/mapping_mat_w2.npy')
        elif lang == 'zh2':
            loadData = np.load('../saved_model/ent_embeds_zh2.npy')
            r_embed = np.load('../saved_model/rel_embeds_zh2.npy')
            mapping = np.load('../saved_model/mapping_mat_zh2.npy')
        elif lang == 'zh_2':
            loadData = np.load('../saved_model/ent_embeds_zh_2.npy')
            r_embed = np.load('../saved_model/rel_embeds_zh_2.npy')
            mapping = np.load('../saved_model/mapping_mat_zh_2.npy')
        elif lang == 'ja2':
            loadData = np.load('../saved_model/ent_embeds_ja_2.npy')
            r_embed = np.load('../saved_model/rel_embeds_ja_2.npy')
            mapping = np.load('../saved_model/mapping_mat_ja_2.npy')
        elif lang == 'fr2':
            loadData = np.load('../saved_model/ent_embeds_fr_2.npy')
            r_embed = np.load('../saved_model/rel_embeds_fr_2.npy')
            mapping = np.load('../saved_model/mapping_mat_fr_2.npy')

        # exit(0)
        # test_embeds1_mapped = np.matmul(embeds1, mapping)
        self.mapping = torch.Tensor(mapping)
        self.G_dataset = G_dataset
        self.embed = torch.Tensor(loadData)
        # print(r_embed.shape)
        # exit(0)
        # self.embed = torch.Tensor(mapping).cuda()
        self.e_sim =self.cosine_matrix(self.embed[:self.split], self.embed[self.split:])
        if self.model_name == 'mean_pooling':
            
            # loadData = np.load('../saved_model/ent_embeds_ja.npy')
            self.embed = torch.Tensor(loadData)
            self.embed[list(range(self.split))] = self.mapping[list(range(self.split))]
            self.e_sim =self.cosine_matrix(self.embed[:self.split], self.embed[self.split:])
            self.e_embed = self.embed
            self.r_embed = torch.Tensor(r_embed)
            # self.r_embed = self.proxy_r()
            self.get_r_map(lang)
            # self.L = torch.matmul(self.embed[list(range(19388))], self.mapping)
            self.L = self.mapping[list(range(self.split))]
            self.R = self.embed[list(range(self.split, self.embed.shape[0]))]
        else:
            self.base_model = model
            self.base_model.eval()
            for name, param in self.base_model.named_parameters():
                param.requires_grad = False
        
        self.Lvec = Lvec
        self.Rvec = Rvec
        if self.Lvec is not None:
            self.Lvec.requires_grad = False
            self.Rvec.requires_grad = False
        self.test_indices = test_indices
        self.args = args
        self.test_kgs = copy.deepcopy(self.G_dataset.kgs)
        self.test_kgs_no = copy.deepcopy(self.G_dataset.kgs)
        self.test_indices = test_indices
        self.test_pair = G_dataset.test_pair
        self.train_pair = G_dataset.train_pair
        self.model_pair = G_dataset.model_pair
        self.model_link = G_dataset.model_link
        self.train_link = G_dataset.train_link
        self.test_link = G_dataset.test_link
        self.args = args
        self.test_kgs = copy.deepcopy(self.G_dataset.kgs)
        self.test_kgs_no = copy.deepcopy(self.G_dataset.kgs)
        self.evaluator = evaluator
        self.evaluator = evaluator

    def proxy(self):
        ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val,triple_size = self.G_dataset.reconstruct_test(self.G_dataset.kgs)
        adj = torch.sparse_coo_tensor(indices=adj_list, values=torch.ones_like(adj_list[0, :], dtype=torch.float),
                                      size=[node_size, node_size])
        adj = torch.sparse.softmax(adj, dim=1)
        res_embed = torch.sparse.mm(adj, self.embed)
        kg1_test_entities = self.G_dataset.test_pair[:, 0]
        kg2_test_entities = self.G_dataset.test_pair[:, 1]
        Lvec = res_embed[kg1_test_entities].cpu()
        Rvec = res_embed[kg2_test_entities].cpu()
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        self.evaluator.test_rank(Lvec, Rvec)


    def proxy_r(self):
        r_list = defaultdict(list)

        for (h, r, t) in self.G_dataset.kg1:
            r_list[int(r)].append([int(h), int(t)])
        for (h, r, t) in self.G_dataset.kg2:
            r_list[int(r)].append([int(h), int(t)])
        
        r_embed = torch.Tensor(len(self.G_dataset.rel), self.embed.shape[1])
        for i in range(r_embed.shape[0]):
            cur_ent = torch.Tensor(r_list[i]).reshape(2,-1)
            h = self.embed[cur_ent[0].long()]
            t = self.embed[cur_ent[1].long()]
            r_embed[i] = (h - t).mean(dim=0)
        return r_embed


    def get_r_map(self, lang):
        '''
        self.r_sim_l =self.cosine_matrix(self.r_embed[:self.splitr], self.r_embed[self.splitr:])
        self.r_sim_r =self.cosine_matrix(self.r_embed[self.splitr:], self.r_embed[:self.splitr])
        rankl = (-self.r_sim_l).argsort()
        rankr = (-self.r_sim_r).argsort()
        self.r_map1 = {}
        self.r_map2 = {}
        for i in range(rankl.shape[0]):
            self.r_map1[i] = rankl[i][0] + self.splitr
            print(self.G_dataset.r_dict[i], self.G_dataset.r_dict[int(rankl[i][0] + self.splitr)])
        for i in range(rankr.shape[0]):
            self.r_map2[i + self.splitr] = rankr[i][0]
            print(self.G_dataset.r_dict[i + self.splitr], self.G_dataset.r_dict[int(rankr[i][0])])
        exit(0)
        
        self.r_map1 = {}
        self.r_map2 = {}
        
        
        for i in range(self.splitr):
            cur1 = self.G_dataset.r_dict[i]
            for j in range(self.splitr, len(self.G_dataset.r_dict)):
                cur2 = self.G_dataset.r_dict[j]
                if cur1.split('/')[-1] == cur2.split('/')[-1]:
                    self.r_map1[self.G_dataset.id_r[cur1]] = self.G_dataset.id_r[cur2]
                    self.r_map2[self.G_dataset.id_r[cur2]] = self.G_dataset.id_r[cur1]
                    # print(cur1, cur2)
        '''
        if lang == 'de':
            self.r_map1 = {}
            self.r_map2 = {}
            for i in range(self.splitr):
                self.r_map1[i] = i
                self.r_map2[i] = i
        elif lang == 'y':
            self.r_map1 = {}
            self.r_map2 = {}
            
            '''
            pair1 = set()
            pair2 = set()
            
            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]
                cur_sim = 0
                ans = None
                for j in range(self.splitr, len(self.G_dataset.r_dict)):
                    cur2 = self.G_dataset.r_dict[j]
                    # if cur1.split('/')[-1] == cur2:
                    if difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio() > cur_sim:
                        cur_sim = difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio()
                        ans = cur2
                        # self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                        # self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
                        # print(cur1, cur2)
                pair1.add((cur1, ans))
            
            for i in range(self.splitr, len(self.G_dataset.r_dict)):
                cur2 = self.G_dataset.r_dict[i]
                cur_sim = 0
                ans = None
                for j in range(self.splitr):
                    cur1 = self.G_dataset.r_dict[j]
                    # if cur1.split('/')[-1] == cur2:
                    if difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio() > cur_sim:
                        cur_sim = difflib.SequenceMatcher(None, cur1.split('/')[-1], cur2).quick_ratio()
                        ans = cur1
                        # self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                        # self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
                        # print(cur1, cur2)
                pair2.add((ans, cur2))

            pair = pair1 & pair2

            for p in pair:
                cur1 = p[0]
                cur2 = p[1]
                self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
            '''

            with open('../datasets/D_Y/rel_links') as f:
                lines = f.readlines()
                for line in lines:
                    cur = line.strip().split(' ')
                    self.r_map1[int(cur[0])] = int(cur[1])
                    self.r_map2[int(cur[1])] = int(cur[0])
            

            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]            
                if self.G_dataset.id_r[cur1] not in self.r_map1:
                    self.r_map1[self.G_dataset.id_r[cur1]] = None
            for i in range(self.splitr, len(self.G_dataset.id_r)):
                cur2 = self.G_dataset.r_dict[i]
                if self.G_dataset.id_r[cur2] not in self.r_map2:
                    self.r_map2[self.G_dataset.id_r[cur2]] = None
        elif lang == 'w' or lang == 'w2':
            self.r_map1 = {}
            self.r_map2 = {}
            with open('../datasets/D_W/rel_links') as f:
                lines = f.readlines()
                for line in lines:
                    cur = line.strip().split('\t')
                    self.r_map1[int(cur[0])] = int(cur[1])
                    self.r_map2[int(cur[1])] = int(cur[0])
            

            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]            
                if self.G_dataset.id_r[cur1] not in self.r_map1:
                    self.r_map1[self.G_dataset.id_r[cur1]] = None
            for i in range(self.splitr, len(self.G_dataset.id_r)):
                cur2 = self.G_dataset.r_dict[i]
                if self.G_dataset.id_r[cur2] not in self.r_map2:
                    self.r_map2[self.G_dataset.id_r[cur2]] = None
        else:
            self.r_map1 = {}
            self.r_map2 = {}
            
            
            for i in range(self.splitr):
                cur1 = self.G_dataset.r_dict[i]
                for j in range(self.splitr, len(self.G_dataset.r_dict)):
                    cur2 = self.G_dataset.r_dict[j]
                    if cur1.split('/')[-1] == cur2.split('/')[-1]:
                        self.r_map1[int(self.G_dataset.id_r[cur1])] = int(self.G_dataset.id_r[cur2])
                        self.r_map2[int(self.G_dataset.id_r[cur2])] = int(self.G_dataset.id_r[cur1])
                        # print(cur1, cur2)
                if self.G_dataset.id_r[cur1] not in self.r_map1:
                    self.r_map1[self.G_dataset.id_r[cur1]] = None
            for i in range(self.splitr, len(self.G_dataset.id_r)):
                cur2 = self.G_dataset.r_dict[i]
                if self.G_dataset.id_r[cur2] not in self.r_map2:
                    self.r_map2[self.G_dataset.id_r[cur2]] = None
    
    def explain_EA(self, method, thred=0, num=15,  version = ''):
        self.version = version
        if method == 'EG':
            if self.lang == 'zh':
                if self.version == 1:
                    with open('../datasets/dbp_z_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_z_e/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_z_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/dbp_z_e/exp_ours', str(version))
            elif self.lang == 'zh2':
                if self.version == 1:
                    # with open('../datasets/dbp_z_e_wrong/exp_ours', 'w') as f:
                    with open('../datasets/z_e_2/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        # self.get_test_file_mask('../datasets/dbp_z_e_wrong/exp_ours', str(version))
                        self.get_test_file_mask('../datasets/z_e_2/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_z_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/dbp_z_e/exp_ours', str(version))
            elif self.lang == 'zh_2':
                if self.version == 1:
                    with open('../datasets/dbp_z_e_wrong/exp_ours', 'w') as f:
                    # with open('../datasets/z_e_2/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_z_e_wrong/exp_ours', str(version))
                        # self.get_test_file_mask('../datasets/z_e_2/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_z_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/dbp_z_e/exp_ours', str(version))
            elif self.lang == 'ja2':
                if self.version == 1:
                    with open('../datasets/dbp_j_e_wrong/exp_ours', 'w') as f:
                    # with open('../datasets/z_e_2/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_j_e_wrong/exp_ours', str(version))
                        # self.get_test_file_mask('../datasets/z_e_2/exp_ours', str(version))
            elif self.lang == 'fr2':
                if self.version == 1:
                    with open('../datasets/dbp_f_e_wrong/exp_ours', 'w') as f:
                    # with open('../datasets/z_e_2/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_f_e_wrong/exp_ours', str(version))
                        # self.get_test_file_mask('../datasets/z_e_2/exp_ours', str(version))
            elif self.lang == 'ja':
                if self.version == 1:
                    with open('../datasets/dbp_j_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_j_e/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_j_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/dbp_j_e/exp_ours', str(version))
            elif self.lang == 'fr':
                if self.version == 1:
                    with open('../datasets/dbp_f_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/dbp_f_e/exp_ours', str(version))
                else:
                    with open('../datasets/dbp_f_e/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/dbp_f_e/exp_ours', str(version))
            elif self.lang == 'y':
                if self.version == 1:
                    with open('../datasets/D_Y/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        self.get_test_file_mask('../datasets/D_Y/exp_ours', str(version))
                else:
                    with open('../datasets/D_Y/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/D_Y/exp_ours', str(version))
            elif self.lang == 'w':
                if self.version == 1:
                    with open('../datasets/D_W/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1[:3]:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2[:4]:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    self.get_test_file_mask('../datasets/D_W/exp_ours', str(version))
                else:
                    with open('../datasets/D_W/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/D_W/exp_ours', str(version))
            elif self.lang == 'w2':
                if self.version == 1:
                    with open('../datasets/D_W2/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                        
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            for cur in tri1[:3]:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2[:4]:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    self.get_test_file_mask('../datasets/D_W2/exp_ours', str(version))
                else:
                    with open('../datasets/D_W2/exp_ours', 'w') as f:
                        for gid1, gid2 in self.test_indices:
                            gid1 = int(gid1)
                            gid2 = int(gid2)
                            tri1, tri2, _ = self.explain_ours4(gid1, gid2)
                            exp_tri1, exp_tri2 = self.explain_ours5(gid1, gid2)
                            tri1 = set(tri1 + exp_tri1)
                            tri2 = set(tri2 + exp_tri2)
                            for cur in tri1:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                            for cur in tri2:
                                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    
                        self.get_test_file_mask_two('../datasets/D_W2/exp_ours', str(version))

        elif method == 'shapley':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/dbp_z_e/exp_shapley', str(version), method)
            elif self.lang == 'zh2':
                with open('../datasets/z_e_2/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/z_e_2/exp_shapley', str(version), method)
            elif self.lang == 'zh_2':
                with open('../datasets/dbp_z_e_wrong/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/dbp_z_e_wrong/exp_shapley', str(version), method)
            elif self.lang == 'ja2':
                with open('../datasets/dbp_j_e_wrong/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/dbp_j_e_wrong/exp_shapley', str(version), method)
            elif self.lang == 'fr2':
                with open('../datasets/dbp_f_e_wrong/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/dbp_f_e_wrong/exp_shapley', str(version), method)
            elif self.lang == 'zh2':
                with open('../datasets/z_e_2/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/z_e_2/exp_shapley', str(version), method)
            elif self.lang == 'zh2':
                with open('../datasets/z_e_2/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                       
                        
                self.get_test_file_mask('../datasets/z_e_2/exp_shapley', str(version), method)
            
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/dbp_j_e/exp_shapley', str(version), method)
            elif self.lang == 'w':
                with open('../datasets/D_W/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/D_W/exp_shapley', str(version), method)
            elif self.lang == 'y':
                with open('../datasets/D_Y/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/D_Y/exp_shapley', str(version), method)
            elif self.lang == 'w2':
                with open('../datasets/D_W2/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/D_W2/exp_shapley', str(version), method)
            else:
                with open('../datasets/dbp_f_e/exp_shapley', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        
                        
                self.get_test_file_mask('../datasets/dbp_f_e/exp_shapley', str(version), method)
        
        elif method == 'lime':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_lime(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                        
                        
                        
                self.get_test_file_mask('../datasets/dbp_z_e/exp_lime', str(version), method)
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_lime(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                        
                self.get_test_file_mask('../datasets/dbp_j_e/exp_lime', str(version), method)
            elif self.lang == 'fr':
                with open('../datasets/dbp_f_e/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_lime(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
           
                        
                self.get_test_file_mask('../datasets/dbp_f_e/exp_lime', str(version), method)
            elif self.lang == 'w':
                with open('../datasets/D_W/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/D_W/exp_lime', str(version), method)
            elif self.lang == 'y':
                with open('../datasets/D_Y/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/D_Y/exp_lime', str(version), method)
            elif self.lang == 'zh_2':
                with open('../datasets/dbp_z_e_wrong/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/dbp_z_e_wrong/exp_lime', str(version), method)
            elif self.lang == 'ja2':
                with open('../datasets/dbp_j_e_wrong/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/dbp_j_e_wrong/exp_lime', str(version), method)
            elif self.lang == 'fr2':
                with open('../datasets/dbp_f_e_wrong/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/dbp_f_e_wrong/exp_lime', str(version), method)
            elif self.lang == 'w2':
                with open('../datasets/D_W2/exp_lime', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        tri = self.explain_shapely(gid1, gid2)
                        
                        for cur in tri:
                            f.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                  
                self.get_test_file_mask('../datasets/D_W2/exp_lime', str(version), method)
        elif method == 'anchor':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')             
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
                  
                    
            elif self.lang == 'fr':
                with open('../datasets/dbp_f_e/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
           

            elif self.lang == 'w':
                with open('../datasets/D_W/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
                  

            elif self.lang == 'y':
                with open('../datasets/D_Y/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
            elif self.lang == 'zh_2':
                with open('../datasets/dbp_z_e_wrong/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n') 
            elif self.lang == 'ja2':
                with open('../datasets/dbp_j_e_wrong/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n') 
            elif self.lang == 'fr2':
                with open('../datasets/dbp_f_e_wrong/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n') 
            elif self.lang == 'w2':
                with open('../datasets/D_W2/exp_anchor', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n') 
        
        elif method == 'lore':
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')             
            elif self.lang == 'ja':
                with open('../datasets/dbp_j_e/exp_lore', 'a') as f:
                    for i in trange(114, len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')  
                  
                    
            elif self.lang == 'fr':
                with open('../datasets/dbp_f_e/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')  
           

            elif self.lang == 'w':
                with open('../datasets/D_W/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')  
            elif self.lang == 'w2':
                with open('../datasets/D_W2/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n')
                  

            elif self.lang == 'y':
                with open('../datasets/D_Y/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_lore(gid1, gid2)
                        f.write(str(rule) + '\n') 
            elif self.lang == 'zh_2':
                with open('../datasets/dbp_z_e_wrong/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n') 
            elif self.lang == 'ja2':
                with open('../datasets/dbp_j_e_wrong/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n') 
            elif self.lang == 'fr2':
                with open('../datasets/dbp_f_e_wrong/exp_lore', 'w') as f:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        rule = self.explain_anchor(gid1, gid2)
                        f.write(str(rule) + '\n')  
        elif method == 'llm':
            
            if self.lang == 'zh':
                with open('../datasets/dbp_z_e/exp_llm_per_prompt', 'w') as f, open('../datasets/dbp_z_e/exp_llm_per_feature', 'w') as f1:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        Z, Y, p1, p2 = self.explain_llm(gid1, gid2)
                        feature_name =''
                        for i in range(len(p1) + len(p2) - 1):
                            feature_name += 'f' + str(i) +','
                        feature_name += 'f' + str(len(p1) + len(p2) - 1)
                        context = 'We have a machine learning model that predicts based on {} features: [{}]. The model has been trained on a dataset and has made the following predictions\n'.format(len(p1) + len(p2), feature_name)
                        dataset = 'Dataset:\n'
                        for i in range(len(Z)):
                            Input = 'Input:'
                            for j in range(len(Z[i])):
                                Input += 'f' + str(j) + '=' + str(int(Z[i][j])) +', '
                            Output = 'Output: {}'.format(Y[i])
                            dataset += Input + '\n'
                            dataset += Output + '\n'
                        Qestions = 'Based on the model\'s predictions and the given dataset, please rank features according to their importance in determining the model\'s prediction?\n'
                        # Instructions = 'Instructions: Think about the question. After explaining your reasoning, provide your answer as features ranked from most important to least important, in descending order. Only provide the feature names on the last line. Do not provide any further details on the last line.\n'
                        Instructions = 'Instructions: Only output the ranking feature names.\n'
                        prompt = context + dataset + Qestions + Instructions + '-------------------------\n'
                        f.write(prompt + '\n')                
                        for i in range(len(p1)):
                            f1.write(str(p1[i][0]) + ',' + str(p1[i][1]) + ',' + str(p1[i][2]) + '\t')
                        for i in range(len(p2)):
                            f1.write(str(p2[i][0]) + ',' + str(p2[i][1]) + ',' + str(p2[i][2]) + '\t')
                        f1.write('\n')
            elif self.lang == 'w':
                with open('../datasets/D_W/exp_llm_per_prompt', 'w') as f, open('../datasets/D_W/exp_llm_per_feature', 'w') as f1:
                    for i in trange(len(self.test_indices)):
                        gid1, gid2 = self.test_indices[i]
                        gid1 = int(gid1)
                        gid2 = int(gid2)
                        # exp = self.explain(gid1, gid2)
                        Z, Y, p1, p2 = self.explain_llm(gid1, gid2)
                        feature_name =''
                        for i in range(len(p1) + len(p2) - 1):
                            feature_name += 'f' + str(i) +','
                        feature_name += 'f' + str(len(p1) + len(p2) - 1)
                        context = 'We have a machine learning model that predicts based on {} features: [{}]. The model has been trained on a dataset and has made the following predictions\n'.format(len(p1) + len(p2), feature_name)
                        dataset = 'Dataset:\n'
                        for i in range(len(Z)):
                            Input = 'Input:'
                            for j in range(len(Z[i])):
                                Input += 'f' + str(j) + '=' + str(int(Z[i][j])) +', '
                            Output = 'Output: {}'.format(Y[i])
                            dataset += Input + '\n'
                            dataset += Output + '\n'
                        Qestions = 'Based on the model\'s predictions and the given dataset, please rank features according to their importance in determining the model\'s prediction?\n'
                        # Instructions = 'Instructions: Think about the question. After explaining your reasoning, provide your answer as features ranked from most important to least important, in descending order. Only provide the feature names on the last line. Do not provide any further details on the last line.\n'
                        Instructions = 'Instructions: Only output the ranking feature names.\n'
                        prompt = context + dataset + Qestions + Instructions + '-------------------------\n'
                        f.write(prompt + '\n')                
                        for i in range(len(p1)):
                            f1.write(str(p1[i][0]) + ',' + str(p1[i][1]) + ',' + str(p1[i][2]) + '\t')
                        for i in range(len(p2)):
                            f1.write(str(p2[i][0]) + ',' + str(p2[i][1]) + ',' + str(p2[i][2]) + '\t')
                        f1.write('\n')

        
        elif method == 'phase1':
            with open('../datasets/dbp_z_e/exp_ours_phase1' + str(version), 'w') as f:
                for i in trange(len(self.model_pair)):
                    gid1, gid2 = self.model_pair[i]
                    # if self.test_link[str(gid1)] == str(gid2):
                        # continue
                    gid1 = int(gid1)
                    gid2 = int(gid2)
                    # exp = self.explain(gid1, gid2)
                    tri1, tri2, pair = self.explain_ours4(gid1, gid2)
                    judge = 0
                    if self.test_link[str(gid1)] == str(gid2):
                        judge = 1
                    f.write('wrong---------------------\n')
                    f.write(self.G_dataset.ent_dict[gid1] + '\t' + self.G_dataset.ent_dict[gid2] + '\t' + str(judge) + '\n')
                    for cur in pair:
                        if str(cur[0]) in self.test_link:
                            if self.test_link[str(cur[0])] == str(cur[1]):
                                judge = 1
                            else:
                                judge = 0
                            f.write(self.G_dataset.ent_dict[cur[0]] + '\t' + self.G_dataset.ent_dict[cur[1]] + '\t' + str(judge) +'\n')
                        else:
                            f.write(self.G_dataset.ent_dict[cur[0]] + '\t' + self.G_dataset.ent_dict[cur[1]] +'\n')
                    for k in range(len(tri1)):
                        cur1 = tri1[k]
                        cur2 = tri2[k]
                        f.write(self.G_dataset.ent_dict[cur1[0]] + '\t' + self.G_dataset.r_dict[cur1[1]] + '\t' + self.G_dataset.ent_dict[cur1[2]] + '\n')
                        f.write(self.G_dataset.ent_dict[cur2[0]] + '\t' + self.G_dataset.r_dict[cur2[1]] + '\t' + self.G_dataset.ent_dict[cur2[2]] + '\n')
                        f.write('------------------------------------------\n')
                    tri1, tri2, pair = self.explain_ours4(gid1, int(self.test_link[str(gid1)]))
                    judge = 0
                    if self.test_link[str(gid1)] == str(gid2):
                        judge = 1
                    f.write('right---------------------\n')
                    f.write(self.G_dataset.ent_dict[gid1] + '\t' + self.G_dataset.ent_dict[int(self.test_link[str(gid1)])] + '\t' + str(1) + '\n')
                    for cur in pair:
                        if str(cur[0]) in self.test_link:
                            if self.test_link[str(cur[0])] == str(cur[1]):
                                judge = 1
                            else:
                                judge = 0
                            f.write(self.G_dataset.ent_dict[cur[0]] + '\t' + self.G_dataset.ent_dict[cur[1]] + '\t' + str(judge) +'\n')
                        else:
                            f.write(self.G_dataset.ent_dict[cur[0]] + '\t' + self.G_dataset.ent_dict[cur[1]] +'\n')
                    for k in range(len(tri1)):
                        cur1 = tri1[k]
                        cur2 = tri2[k]
                        f.write(self.G_dataset.ent_dict[cur1[0]] + '\t' + self.G_dataset.r_dict[cur1[1]] + '\t' + self.G_dataset.ent_dict[cur1[2]] + '\n')
                        f.write(self.G_dataset.ent_dict[cur2[0]] + '\t' + self.G_dataset.r_dict[cur2[1]] + '\t' + self.G_dataset.ent_dict[cur2[2]] + '\n')
                        f.write('------------------------------------------\n')
                    f.write('**********************************\n')
        elif method == 'confidence':
            K = 100
            r1_func, r1_func_r, r2_func, r2_func_r = self.get_r_func()
            node = {}
            ground = {}
            cur_link = {}
            cur_pair = set()
            for p in self.model_pair:
                cur_link[int(p[0])] = int(p[1])
                cur_pair.add((int(p[0]), int(p[1])))
            for cur in self.test_pair:
                ground[str(cur[0])] = str(cur[1])
                ground[str(cur[1])] = str(cur[0])
            node_set = defaultdict(float)
            
            kg2 = set()
            all_kg1 = set()
            for p in self.test_pair:
                kg2.add(p[1])
                all_kg1.add(p[0])
            ans_pair = set()
            for cur in self.test_link:
                ans_pair.add((int(cur), int(self.test_link[str(cur)])))
            
            kg1_embed = self.embed[:self.split]
            kg2_list = list(kg2)
            kg2_list.sort()
            kg2_embed = self.embed[kg2_list]
            sim = self.cosine_matrix(kg1_embed, kg2_embed)
            rank = (-sim).argsort()
            # 示例数据
            

            x = []
            y = []
            r = 0
            w = 0
            count = 0
            for cur in cur_link:
                gid1 = cur
                gid2 = cur_link[cur]
                pair, score= self.get_pair_score(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                # score2 = self.get_pair_score_high(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                # score +=  score2
                if cur_link[cur] == int(self.test_link[str(cur)]):
                    x.append(1)
                    y.append(score)
                    r += score
                    # r += sim[int(cur), cur_link[cur] - 19388]
                    count += 1
                else:
                    x.append(0)
                    y.append(score)
                    w += score
                    # w += sim[int(cur), cur_link[cur] - 19388]
            print(r / count, w / (len(x) - count))
            plt.scatter(x, y)

            # 添加标题和标签
            plt.title('2D Scatter Plot')
            plt.xlabel('X Values')
            plt.ylabel('Y Values')

            # 显示图形
            plt.show()
            plt.savefig('confidence.png')


                    
        
        elif method == 'repair':
            K = 100
            r1_func, r1_func_r, r2_func, r2_func_r = self.get_r_func()
            node = {}
            ground = {}
            cur_link = {}
            cur_pair = set()
            for p in self.model_pair:
                cur_link[int(p[0])] = int(p[1])
                cur_pair.add((int(p[0]), int(p[1])))
            for cur in self.test_pair:
                ground[str(cur[0])] = str(cur[1])
                ground[str(cur[1])] = str(cur[0])
            node_set = defaultdict(float)
            
            kg2 = set()
            all_kg1 = set()
            for p in self.test_pair:
                kg2.add(p[1])
                all_kg1.add(p[0])
            ans_pair = set()
            for cur in self.test_link:
                ans_pair.add((int(cur), int(self.test_link[str(cur)])))
            
            kg1_embed = self.embed[:self.split]
            kg2_list = list(kg2)
            kg2_list.sort()
            kg2_embed = self.embed[kg2_list]
            sim = self.cosine_matrix(kg1_embed, kg2_embed)
            rank = (-sim).argsort()
            for cur in cur_link:
                gid1 = cur
                gid2 = cur_link[cur]
                pair, score= self.get_pair_score(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                # score2 = self.get_pair_score_high(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                # score +=  score2
                if len(pair) > 0:
                    node[str(gid1) + '\t' + str(gid2)] = pair
                    node_set[str(gid1) + '\t' + str(gid2)] = score
            T1 = time.clock()
            
            c_set, new_model_pair, count1 = self.conflict_count(self.model_pair)
            
            kg1, _, cur_pair = self.conflict_solve(c_set, node_set, new_model_pair, kg2, count1, ground)
            while(len(kg1) > 0):
                cur_link = {}
                cur_link_r = {}
                print(len(cur_pair))
                print(len(kg1))
                cur_kg1 = set()
                cur_kg2 = set()
                for p in cur_pair:
                    cur_link[int(p[0])] = int(p[1])
                    cur_kg1.add(p[0])
                    cur_kg2.add(p[1])
                    cur_link_r[int(p[1])] = int(p[0])
                print(len(cur_kg1), len(cur_kg2), len(all_kg1 - cur_kg1), len(kg2 - cur_kg2))
                last_len = len(kg1)
                # _, kg1 = self.adjust_conflict(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, 100, conflict_link)
                _, kg1 = self.adjust_rank(kg1, kg2_list, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, 5, rank)
                if len(kg1) >= last_len:
                    break
                print(len(cur_pair))
                # print(len(delete_pair & ans_pair), len(delete_pair) )
                print(len(cur_pair & ans_pair) / len(ans_pair))
            
            
            # find low confidence conflict
            print('start low confidence conflict solving')
            last_len1 = None
            while True:
                cur_link = {}
                cur_link_r = {}
                kg1 = set()
                print(len(cur_pair & ans_pair) / len(ans_pair))
                for p in cur_pair:
                    cur_link[int(p[0])] = int(p[1])
                    kg1.add(int(p[0]))
                    cur_link_r[int(p[1])] = int(p[0])
                kg1 = all_kg1 - kg1
                
                for cur in cur_link:
                    gid1 = cur
                    gid2 = cur_link[cur]
                    _, score = self.find_low_confidence(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    # _, score = self.get_pair_score6(gid1, gid2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    if score <= 0.7:
                        kg1.add(gid1)
                        cur_pair.remove((gid1, gid2))
                if last_len1 != None and len(kg1) >= last_len1:
                    break
                else:
                    last_len1 = len(kg1)
                print(len(cur_pair & ans_pair) / len(ans_pair))
                # while(len(kg1) > 0):
                cur_link = {}
                cur_link_r = {}
                print(len(cur_pair))
                print(len(kg1))
                cur_kg1 = set()
                cur_kg2 = set()
                for p in cur_pair:
                    cur_link[int(p[0])] = int(p[1])
                    cur_kg1.add(p[0])
                    cur_kg2.add(p[1])
                    cur_link_r[int(p[1])] = int(p[0])
                print(len(cur_kg1), len(cur_kg2), len(all_kg1 - cur_kg1), len(kg2 - cur_kg2))
                # last_len = len(kg1)
                # _, kg1 = self.adjust_conflict(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, 100, conflict_link)
                _, kg1 = self.adjust_no_explain(kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, 1)
                # if len(kg1) >= last_len:
                    # break
                print(len(cur_pair))
                # print(len(delete_pair & ans_pair), len(delete_pair) )
                print(len(cur_pair & ans_pair) / len(ans_pair))
            
            print('repair result :', len(cur_pair & ans_pair) / len(ans_pair))
            solve_kg1 = set()
            solve_kg2 = set()
            for p in cur_pair:
                solve_kg1.add(p[0])
                solve_kg2.add(p[1])
            print(len(solve_kg1), len(solve_kg2))
            left_kg1 = all_kg1 - solve_kg1
            left_kg2 = kg2 - solve_kg2
            print(len(left_kg1), len(left_kg2))
            new_pair, _ = self.re_align(left_kg1, left_kg2)
            cur_pair |= new_pair
            print('final res: ', len(cur_pair & ans_pair) / len(ans_pair))
            with open('repair_pair', 'w') as f:
                for p in cur_pair:
                    f.write(str(p[0]) + '\t' + str(p[1]) + '\n')
            T2 = time.clock()
            print(T2 - T1)


    def conflict_solve(self, c_set, node_set, new_model_pair, kg2, count1, ground):
        count = 0
        kg1 = set()
        cur_kg2 = set()
        cur_kg1 = set()
        cur_pair = set()
        for ent in c_set:
            cur_kg2.add(ent)
            if len(c_set[ent]) > 1:
                tmp = 0
                judge = 0
                score = -1e5
                max_e = None

                for e in c_set[ent]:
                    cur_score = node_set[str(e) + '\t' + str(ent)] # +  0.5 * self.e_sim[e, ent - self.split]
                    # cur_score =    self.e_sim[e, ent - self.split]
                    if cur_score >= score:
                        score = cur_score
                        max_e = e
                    '''
                    if node_set[str(e) + '\t' + str(ent)] >= score:
                        score = node_set[str(e) + '\t' + str(ent)]
                        max_e = e
                    
                    if self.e_sim[e, ent - self.split] >= score:
                        score = self.e_sim[e, ent - self.split]
                        max_e = e
                    '''
                new_model_pair[max_e] = ent
                cur_pair.add((max_e, ent))
                for e in c_set[ent]:
                    if e != max_e:
                        kg1.add(e)
                    cur_kg1.add(e)
                if max_e == int(ground[str(ent)]):
                    count += 1
            else:
                for e in c_set[ent]:
                    cur_pair.add((e, ent))

        new_kg2 = kg2 - cur_kg2
        
        print(len(kg1), len(new_kg2), len(cur_kg1), len(cur_kg2), count - count1)
        return kg1, new_kg2, cur_pair

    def adjust_no_explain(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, rule_type):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        
        new_kg1 = set()
        new_pair = set()
        delete_pair = set()
        

        for i in range(len(kg1)):
            candidate = self.candidate_ent(kg1[i], cur_link)
            # kg1_embed = self.embed[[kg1[i]]]
            # kg2_embed = self.embed[candidate]
            # cur_sim = self.cosine_matrix(kg1_embed, kg2_embed)
            # rank = (-cur_sim).argsort()
            # max_e = None
            score = []
            for ent in candidate:
                _, cur_score = self.get_pair_score5(kg1[i], ent, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                cur_score += 0.5 * self.e_sim[kg1[i], ent - self.split]
                score.append(cur_score)
            values, rank = torch.Tensor(score).sort(descending=True)
            for j in range(min(K, len(rank))):
                # print(len(candidate), rank[0][j], len(rank[0]))
                # print(candidate[int(rank[0][j])])
                e2 = int(candidate[int(rank[j])])
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 in kg2:
                    if e2 not in cur_link_r:   
                        cur_pair.add((kg1[i], e2))
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
                    else:
                        if rule_type == 0:
                            _, cur_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        elif rule_type == 1:
                            cur_score = values[j]
                        
                            _, other = self.get_pair_score6(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            other += 0.5 * self.e_sim[cur_link_r[e2], e2 - self.split]
                        elif rule_type == 2:
                            _, cur_score = self.get_pair_score4(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            _, other = self.get_pair_score4(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        if other < cur_score:
                            # print(kg1[i], cur_link_r[e2])
                            cur_pair.remove((cur_link_r[e2], e2))
                            cur_pair.add((kg1[i], e2))
                            new_kg1.add(cur_link_r[e2])
                            cur_link_r[e2] = kg1[i]
                            cur_link[kg1[i]] = e2
                            
                            break
                # new_pair.add((kg1[i], e2))
            if kg1[i] not in cur_link:
                new_kg1.add(kg1[i])
        print(len(new_kg1))
        # print(new_kg1)
        count = 0
        # for cur in new_kg1:
            # if self.model_link[str(cur)] != self.test_link[str(cur)]:
                # count += 1
        # print(count)
        return new_pair, new_kg1

    def candidate_ent(self, e1, cur_link):
        candidate = set()
        for cur in self.G_dataset.gid[e1]:
            if cur[0] != int(e1) and cur[0] in cur_link:
                for t in self.G_dataset.gid[cur_link[cur[0]]]:
                    if t[0] ==  cur_link[cur[0]]:
                        candidate.add(int(t[2]))
                    else:
                        candidate.add(int(t[0]))
            if cur[0] != int(e1) and str(cur[0]) in self.train_link:
                for t in self.G_dataset.gid[int(self.train_link[str(cur[0])])]:
                    if t[0] ==  int(self.train_link[str(cur[0])]):
                        candidate.add(int(t[2]))
                    else:
                        candidate.add(int(t[0]))
            else:
                if cur[2] in cur_link:
                    for t in self.G_dataset.gid[cur_link[cur[2]]]:
                        if t[0] ==  cur_link[cur[2]]:
                            candidate.add(int(t[2]))
                        else:
                            candidate.add(int(t[0]))
                if str(cur[2]) in self.train_link:
                    for t in self.G_dataset.gid[int(self.train_link[str(cur[2])])]:
                        if t[0] ==  int(self.train_link[str(cur[2])]):
                            candidate.add(int(t[2]))
                        else:
                            candidate.add(int(t[0]))
        candidate = list(candidate)
        candidate.sort()
        return candidate
    def re_align(self, kg1, kg2):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        kg1_embed = self.embed[kg1]
        kg2_embed = self.embed[kg2]
        # print(kg1_embed.shape, kg1)
        sim = self.cosine_matrix(kg1_embed, kg2_embed)
        rank = (-sim).argsort()
        new_pair = set()
        ans_pair = set()
        for cur in kg1:
            ans_pair.add((int(cur), int(self.test_link[str(cur)])))
        for i in range(rank.shape[0]):
            new_pair.add((kg1[i], kg2[rank[i][0]]))
        print(len(new_pair & set(ans_pair)))
        print(len(new_pair & set(ans_pair)) / 10500)
        return new_pair, set(kg2)

    def adjust_rank(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, rule_type, rank):
        
        new_kg1 = set()
        new_pair = set()
        delete_pair = set()
        # print(len(kg1))
        kg1 = list(kg1)
        kg1.sort()
        for i in range(len(kg1)):
            for j in range(K):
                e2 = kg2[rank[kg1[i]][j]]
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 not in cur_link_r:   
                    cur_pair.add((kg1[i], e2))
                    cur_link_r[e2] = kg1[i]
                    cur_link[kg1[i]] = e2
                    
                    break
                else:
                    if rule_type == 0:
                        _, cur_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    elif rule_type == 1:
                        _, cur_score = self.get_pair_score1(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        _, other = self.get_pair_score1(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    elif rule_type == 2:
                        _, cur_score = self.get_pair_score4(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        _, other = self.get_pair_score4(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    elif rule_type == 5:
                        _, cur_score = self.get_pair_score6(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # score2 = self.get_pair_score_high(kg1[i], e2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # cur_score += score2
                        _, other = self.get_pair_score6(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # score2 = self.get_pair_score_high(cur_link_r[e2], e2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # other += score2
                    if other < cur_score:
                        # print(kg1[i], cur_link_r[e2])
                        cur_pair.remove((cur_link_r[e2], e2))
                        cur_pair.add((kg1[i], e2))
                        new_kg1.add(cur_link_r[e2])
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
            if kg1[i] not in cur_link:
                new_kg1.add(kg1[i])    
            
            # new_pair.add((kg1[i], e2))
        print(len(new_kg1))
        # print(new_kg1)
        count = 0
        # for cur in new_kg1:
            # if self.model_link[str(cur)] != self.test_link[str(cur)]:
                # count += 1
        # print(count)
        return new_pair, new_kg1

    def adjust(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, rule_type):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        kg1_embed = self.embed[kg1]
        kg2_embed = self.embed[kg2]
        sim = self.cosine_matrix(kg1_embed, kg2_embed)
        rank = (-sim).argsort()
        new_kg1 = set()
        new_pair = set()
        delete_pair = set()
        # print(len(kg1))
        for i in range(rank.shape[0]):
            for j in range(K):
                e2 = kg2[rank[i][j]]
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 not in cur_link_r:   
                    cur_pair.add((kg1[i], e2))
                    cur_link_r[e2] = kg1[i]
                    cur_link[kg1[i]] = e2
                    
                    break
                else:
                    if rule_type == 0:
                        _, cur_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    elif rule_type == 1:
                        _, cur_score = self.get_pair_score1(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        _, other = self.get_pair_score1(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    elif rule_type == 2:
                        _, cur_score = self.get_pair_score4(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        _, other = self.get_pair_score4(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                    elif rule_type == 5:
                        _, cur_score = self.get_pair_score6(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # score2 = self.get_pair_score_high(kg1[i], e2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # cur_score += score2
                        _, other = self.get_pair_score6(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # score2 = self.get_pair_score_high(cur_link_r[e2], e2,r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # other += score2
                    if other < cur_score:
                        # print(kg1[i], cur_link_r[e2])
                        cur_pair.remove((cur_link_r[e2], e2))
                        cur_pair.add((kg1[i], e2))
                        new_kg1.add(cur_link_r[e2])
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
            if kg1[i] not in cur_link:
                new_kg1.add(kg1[i])    
            
            # new_pair.add((kg1[i], e2))
        print(len(new_kg1))
        # print(new_kg1)
        count = 0
        # for cur in new_kg1:
            # if self.model_link[str(cur)] != self.test_link[str(cur)]:
                # count += 1
        # print(count)
        return new_pair, new_kg1

    

    def adjust1(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, rule_set=None):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        kg1_embed = self.embed[kg1]
        kg2_embed = self.embed[kg2]
        sim = self.cosine_matrix(kg1_embed, kg2_embed)
        rank = (-sim).argsort()
        new_kg1 = set()
        new_pair = set()
        delete_pair = set()
        # print(len(kg1))
        for i in range(rank.shape[0]):
            for j in range(K):
                e2 = kg2[rank[i][j]]
                # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                if e2 not in cur_link_r:   
                    cur_pair.add((kg1[i], e2))
                    cur_link_r[e2] = kg1[i]
                    cur_link[kg1[i]] = e2
                    
                    break
                else:
                    
                    _, cur_score = self.get_pair_score3(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set)
                    # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                    # cur_score =  self.e_sim[kg1[i], e2- self.split]
                    _, other = self.get_pair_score3(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set)
                    # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                    # other = self.e_sim[cur_link_r[e2], e2- self.split]
                    if other < cur_score:
                        # print(kg1[i], cur_link_r[e2])
                        cur_pair.remove((cur_link_r[e2], e2))
                        cur_pair.add((kg1[i], e2))
                        new_kg1.add(cur_link_r[e2])
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
            # new_pair.add((kg1[i], e2))
        print(len(new_kg1))
        # print(new_kg1)
        count = 0
        # for cur in new_kg1:
            # if self.model_link[str(cur)] != self.test_link[str(cur)]:
                # count += 1
        # print(count)
        return new_pair, new_kg1

    def adjust_conflict(self, kg1, kg2, cur_link, cur_link_r, r1_func, r1_func_r, r2_func, r2_func_r, node_set, cur_pair, K, conflict_link):
        kg1 = list(kg1)
        kg2 = list(kg2)
        kg1.sort()
        kg2.sort()
        kg1_embed = self.embed[kg1]
        kg2_embed = self.embed[kg2]
        sim = self.cosine_matrix(kg1_embed, kg2_embed)
        rank = (-sim).argsort()
        new_kg1 = set()
        new_pair = set()
        # print(len(kg1))
        # print(rank.shape[0])
        for i in range(rank.shape[0]):
            # print(kg1[i])
            if kg1[i] in conflict_link:
                _, cur_score = self.get_pair_score(kg1[i], conflict_link[kg1[i]], r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                for j in range(K):
                    e2 = kg2[rank[i][j]]
                    # if e2 != conflict_link[kg1[i]]:
                        # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                    if e2 != conflict_link[kg1[i]]:
                        if e2 not in cur_link_r:   
                            # _, cur_score = self.get_pair_score(kg1[i], conflict_link[kg1[i]], r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            _, other = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            if other > cur_score:
                                cur_pair.add((kg1[i], e2))
                                if (kg1[i], conflict_link[kg1[i]]) in cur_pair:
                                    cur_pair.remove((kg1[i], conflict_link[kg1[i]]))
                                    del cur_link_r[conflict_link[kg1[i]]]
                                    # print('del', conflict_link[kg1[i]])
                                cur_link_r[e2] = kg1[i]
                                cur_link[kg1[i]] = e2
                                
                                break
                        else:
                            
                            _, other_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                            if other_score >= cur_score:
                                # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                                # cur_score =  self.e_sim[kg1[i], e2- self.split]
                                _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                                # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                                # other = self.e_sim[cur_link_r[e2], e2- self.split]
                                if other < other_score:
                                    # print(kg1[i], cur_link_r[e2])
                                    cur_pair.remove((cur_link_r[e2], e2))
                                    if (kg1[i], conflict_link[kg1[i]]) in cur_pair:
                                        cur_pair.remove((kg1[i], conflict_link[kg1[i]]))
                                        del cur_link_r[conflict_link[kg1[i]]]
                                        # print('del', conflict_link[kg1[i]])
                                    cur_pair.add((kg1[i], e2))
                                    new_kg1.add(cur_link_r[e2])
                                    cur_link_r[e2] = kg1[i]
                                    cur_link[kg1[i]] = e2
                                    
                                    break
            else:
                for j in range(K):
                    e2 = kg2[rank[i][j]]
                    # if self.get_r_conflict(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link) == 0:
                    if e2 not in cur_link_r:   
                        cur_pair.add((kg1[i], e2))
                        cur_link_r[e2] = kg1[i]
                        cur_link[kg1[i]] = e2
                        
                        break
                    else:
                        
                        _, cur_score = self.get_pair_score(kg1[i], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # cur_score += 0.5 * self.e_sim[kg1[i], e2- self.split]
                        # cur_score =  self.e_sim[kg1[i], e2- self.split]
                        _, other = self.get_pair_score(cur_link_r[e2], e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
                        # other += 0.5 * self.e_sim[cur_link_r[e2], e2- self.split]
                        # other = self.e_sim[cur_link_r[e2], e2- self.split]
                        if other < cur_score:
                            # print(kg1[i], cur_link_r[e2])
                            cur_pair.remove((cur_link_r[e2], e2))
                            cur_pair.add((kg1[i], e2))
                            new_kg1.add(cur_link_r[e2])
                            cur_link_r[e2] = kg1[i]
                            cur_link[kg1[i]] = e2
                            
                            break
                # new_pair.add((kg1[i], e2))
        print(len(new_kg1))
        # print(new_kg1)
        count = 0
        for cur in new_kg1:
            if self.model_link[str(cur)] != self.test_link[str(cur)]:
                count += 1
        print(count)
        return new_pair, new_kg1
    
    
    def get_pair_score_high(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        exp_tri1 = []
        exp_tri2 = []
        cur_link = self.model_link
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        score = 0
        pair = set()
        for cur in neigh12:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh21:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh21:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        
        r_pair = []
        pair = list(pair)
        r_score = 0
        for k in range(len(pair)):
            p = pair[k]
            # print(self.search_2_hop_tri(e1, p[0]))
            two_hop_list = self.search_2_hop_tri1(e1, p[0])
            
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            index = 0
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r1.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e1]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            
            for i in range(len(pair_r)):
                pr = pair_r[i]
                cur_score = 1
                cur_score1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                if two_hop[pr[0]][0][0] == e1:
                    cur_score1 = min(cur_score1,r1_func_r[str(two_hop[pr[0]][0][1])])
                else:
                    cur_score1 = min(cur_score1,r1_func[str(two_hop[pr[0]][0][1])])
                if two_hop[pr[0]][1][0] == p[0]:
                    cur_score1 = r1_func_r[str(two_hop[pr[0]][1][1])] * cur_score1
                else:
                    cur_score1 = r1_func[str(two_hop[pr[0]][1][1])] * cur_score1

                # r_score = max(r_score, cur_score)
                
            r_score += cur_score * float(self.e_sim[p[0]][p[1] - self.split])

                
        pair = set()
        for cur in neigh11:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh22:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh11:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            two_hop_list = self.search_2_hop_tri1(e2, p[1])
            
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]), dim=0))
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r2.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e2]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []
            for i in range(len(pair_r)):
                cur_score = 1
                cur_score1 = 1
                pr = pair_r[i]
                
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])

                if two_hop[pr[1]][0][0] == e2:
                    cur_score1 = min(cur_score1,r2_func_r[str(two_hop[pr[1]][0][1])])
                else:
                    cur_score1 = min(cur_score1,r2_func[str(two_hop[pr[1]][0][1])])
                if two_hop[pr[1]][1][0] == p[1]:
                    cur_score1 = r2_func_r[str(two_hop[pr[1]][1][1])] * cur_score1
                else:
                    cur_score1 = r2_func[str(two_hop[pr[1]][1][1])] * cur_score1
                # r_score = max(r_score, cur_score)
            r_score += cur_score * float(self.e_sim[p[0]][p[1] - self.split])
        if r_score == 0:
            pair = set()
            for cur in neigh12:
                if str(cur) in cur_link:
                    if int(cur_link[str(cur)]) in neigh22:
                        pair.add((cur, int(cur_link[str(cur)])))
            for cur in neigh12:
                if str(cur) in self.train_link:
                    if int(self.train_link[str(cur)]) in neigh22:
                        pair.add((cur, int(self.train_link[str(cur)])))
            pair = list(pair)
            for k in range(len(pair)):
                p = pair[k]
                r_score += 0.0001 * float(self.e_sim[p[0]][p[1] - self.split])
        else:
            return  set(), sigmoid(0.001 * r_score)
            
        return set(), sigmoid(0.001 * r_score)


    def get_pair_score(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), sigmoid(0)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            for pr in pair_r:
                direct = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                # r_score = max(r_score, cur_score)
                r_score += cur_score
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            

        return pair_node, sigmoid(score)
    
    def get_pair_score1(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            for pr in pair_r:
                direct = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add(map_r2)
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add(map_r1)


                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            for pr in pair_r:
                for cur in neigh_r1:
                    if ((tri1[pr[0]][1], cur) in self.conflict_r_pair or (cur, tri1[pr[0]][1]) in self.conflict_r_pair):
                        r_score = 0
                for cur in neigh_r2:
                    if ((tri2[pr[1]][1], cur) in self.conflict_r_pair or (cur, tri2[pr[1]][1]) in self.conflict_r_pair):
                        r_score = 0
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            

        return pair_node, score
    
    def get_pair_score2(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set1, rule_set2, thred):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        r_pair = []
        pair = list(pair)
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                '''
                for j in range(len(pair_r)):
                    if i != j:
                        rule_set2[(tri1[pair_r[i][0]][1], tri2[pair_r[j][1]][1])].add((e1, e2))
                '''
                # cur_r_pair.append((tri1[pr[0]][1],tri2[pr[1]][1]))
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    cur_score_r = min(cur_score_r,r1_func[str(tri1[pr[0]][1])])
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    cur_score_r = min(cur_score_r,r1_func_r[str(tri1[pr[0]][1])])
                    direct = 1
                    r_1 = (tri1[pr[0]][1], 1)
                
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                    cur_score_r = min(cur_score_r,r2_func[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 0)
                    if direct == 1:
                        judge = 1
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    cur_score_r = min(cur_score_r,r2_func_r[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 1)
                    if direct == 0:
                        judge = 1
                cur_r_pair.append((r_1, r_2))
                if cur_score_r >= thred and judge == 0:
                    # rule_set1[(tri1[pr[0]][1], tri2[pr[1]][1])].add((e1, e2))
                    rule_set1[(r_1, r_2)].add((e1, e2))
                r_score += cur_score
            r_pair.append(cur_r_pair)
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        
        for i in range(len(r_pair) - 1):
            for j in range(i + 1, len(r_pair)):
                for cur1 in r_pair[i]:
                    for cur2 in r_pair[j]:
                        # if str(pair[i][0]) in self.train_link:
                        rule_set2[(cur1[0], cur2[1])].add((e1, e2))
                        rule_set2[(cur2[0], cur1[1])].add((e1, e2))
                        # else:
                            # rule_set2[(cur1[0], cur2[1])].add((pair[i][0], pair[j][1]))
                            # rule_set2[(cur2[0], cur1[1])].add((e1, e2))
                            # rule_set2[(cur2[0], cur1[1])].add((pair[i][0], pair[j][1]))
                        

        return pair_node, score

    def get_pair_score3(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        r_pair = []
        pair = list(pair)
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                '''
                for j in range(len(pair_r)):
                    if i != j:
                        rule_set2[(tri1[pair_r[i][0]][1], tri2[pair_r[j][1]][1])].add((e1, e2))
                '''
                # cur_r_pair.append((tri1[pr[0]][1],tri2[pr[1]][1]))
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    
                    direct = 1
                    r_1 = (tri1[pr[0]][1], 1)
                
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                    
                    r_2 = (tri2[pr[1]][1], 0)
                    if direct == 1:
                        judge = 1
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                   
                    r_2 = (tri2[pr[1]][1], 1)
                    if direct == 0:
                        judge = 1
                cur_r_pair.append((r_1, r_2))
                # if (r_1, r_2) in rule_set:
                cur_score = 1
                r_score += cur_score
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        

                        

        return pair_node, score

    def get_pair_score4(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link, rule_set=None):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
            # return self.get_high_order_pair_score(e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        score_list = []
        score = 0
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])

                    r_1 = (tri1[pr[0]][1], 1)
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])       
                    r_2 = (tri2[pr[1]][1], 0)
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 1)
                
                # map_r2 = None
                # map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                    # map_r2 = int(self.r_map2[tri2[pr[1]][1]])
                
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add(map_r2)
                # if tri1[pr[0]][1] in self.r_map1:
                    # print(self.r_map1[tri1[pr[0]][1]])
                    # map_r1 = int(self.r_map1[tri1[pr[0]][1]])
                map_r1 = self.r_map1[tri1[pr[0]][1]]

                neigh_r2.add(map_r1)
                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                r_score += cur_score
                cur_r_pair.append((r_1, r_2))
            '''
            for pr in pair_r:
                for cur in neigh_r1:
                    if ((tri1[pr[0]][1], cur) in self.conflict_r_pair or (cur, tri1[pr[0]][1]) in self.conflict_r_pair):
                        # print(self.G_dataset.r_dict[tri1[pr[0]][1]], self.G_dataset.r_dict[cur])
                        # print(self.G_dataset.r_dict[cur], self.G_dataset.r_dict[tri1[pr[0]][1]])
                        r_score = 0
                for cur in neigh_r2:
                    if ((tri2[pr[1]][1], cur) in self.conflict_r_pair or (cur, tri2[pr[1]][1]) in self.conflict_r_pair):
                        # print(self.G_dataset.r_dict[tri2[pr[1]][1]],self.G_dataset.r_dict[cur])
                        # print(self.G_dataset.r_dict[cur], self.G_dataset.r_dict[tri2[pr[1]][1]])
                        r_score = 0
            '''
            r_pair.append(cur_r_pair)
            score_list.append(r_score)
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        '''
        for i in range(len(r_pair) - 1):
            for j in range(i + 1, len(r_pair)):
                for cur1 in r_pair[i]:
                    for cur2 in r_pair[j]:
                        cur_sim = self.e_sim[pair[i][0]][pair[j][1] - self.split]
                        # print(cur1[0][1], cur2[1][1], self.r_map1[cur1[0][0]], cur1[0][0], cur2[1][0], self.r_map2[cur2[1][0]])
                        map1 = self.r_map2[cur2[1][0]]
                        map2 = self.r_map1[cur1[0][0]]
                        # print(self.line_id)
                        # exit(0)
                        # print((cur1[0][0], cur1[0][1], map1, cur2[1][1]))
                        if (((cur1[0][0], cur1[0][1]), (map1, cur2[1][1]))) in self.line_id or (((map2, cur1[0][1]), (cur2[1][0], cur2[1][1]))) in self.line_id:
                            # print(self.G_dataset.r_dict[cur1[0][0]], cur1[0][1], self.G_dataset.r_dict[cur2[1][0]], cur2[1][1])
                            # print(self.G_dataset.ent_dict[e1] + '\t' + self.G_dataset.ent_dict[e2], self.test_link[str(e1)] ==  str(e2))
                            score_list.append(0.01)
                            pair.append((pair[i][0], pair[j][1]))
        '''                        
                            
                        
        
        
        for i in range(len(score_list)):
            score += score_list[i] * float(self.e_sim[pair[i][0]][pair[i][1] - self.split])
            # score += score_list[i] 
        return pair_node, score

    def get_pair_score5(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), sigmoid(0)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                # r1.append(self.r_embed[cur[1]])
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]),dim=0))
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
                # r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            pair_r_d = set() 
            for pr in pair_r:
                direct1 = 0
                direct2 = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    direct1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    direct2 = 1
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add((map_r2, direct2))
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add((map_r1,direct1))
                pair_r_d.add(((tri1[pr[0]][1], direct1), (tri2[pr[1]][1], direct2)))

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            for pr in pair_r_d:
                for cur in neigh_r1:
                    if ((pr[0], cur) in self.conflict_id or (cur, pr[0]) in self.conflict_id):
                        r_score = 0
                for cur in neigh_r2:
                    if ((pr[1], cur) in self.conflict_id or (cur, pr[1]) in self.conflict_id):
                        r_score = 0
            # score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            

        return pair_node, sigmoid(score)

    def get_pair_score6(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            # return set(), self.get_pair_score_high(e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
            return set(), sigmoid(0)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                # r1.append(self.r_embed[cur[1]])
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]),dim=0))
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
                # r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            pair_r_d = set() 
            for pr in pair_r:
                direct1 = 0
                direct2 = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score,r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    direct1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    direct2 = 1
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add((map_r2, direct2))
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add((map_r1,direct1))
                pair_r_d.add(((tri1[pr[0]][1], direct1), (tri2[pr[1]][1], direct2)))

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            # print('solve relation alignment conflict----')
            # print(pair_r_d,  neigh_r1, neigh_r2)
            '''
            a = 1
            for pr in pair_r_d:
                for cur in neigh_r1:
                    if ((pr[0], cur) in self.conflict_id or (cur, pr[0]) in self.conflict_id):
                        # print(r1_func_r[str(pr[0][0])], pr[0][0])
                        if pr[0][1] == 1 and r1_func_r[str(pr[0][0])] >= a and cur[1] == 1 and r1_func_r[str(cur[0])] >= a:
                            r_score = 0
                        elif pr[0][1] == 1 and r1_func_r[str(pr[0][0])] >= a and cur[1] == 0 and r1_func[str(cur[0])] >= a:
                            r_score = 0
                        elif pr[0][1] == 0 and r1_func[str(pr[0][0])] >= a and cur[1] == 1 and r1_func_r[str(cur[0])] >= a:
                            r_score = 0
                        elif pr[0][1] == 0 and r1_func[str(pr[0][0])] >= a and cur[1] == 0 and r1_func[str(cur[0])] >= a:
                            r_score = 0
                        # print(self.G_dataset.r_dict[pr[0][0]], pr[0][1], self.G_dataset.r_dict[cur[0]], cur[1])
                        # print('solve')
                        
                for cur in neigh_r2:
                    if ((pr[1], cur) in self.conflict_id or (cur, pr[1]) in self.conflict_id):
                        # print(self.G_dataset.r_dict[pr[1][0]], pr[1][1], self.G_dataset.r_dict[cur[0]], cur[1])
                        
                        
                        # print('solve')
                        if pr[1][1] == 1 and r2_func_r[str(pr[1][0])] >= a and cur[1] == 1 and r2_func_r[str(cur[0])] >= a:
                            r_score = 0
                        elif pr[1][1] == 1 and r2_func_r[str(pr[1][0])] >= a and cur[1] == 0 and r2_func[str(cur[0])] >= a:
                            r_score = 0
                        elif pr[1][1] == 0 and r2_func[str(pr[1][0])] >= a and cur[1] == 1 and r2_func_r[str(cur[0])] >= a:
                            r_score = 0
                        elif pr[1][1] == 0 and r2_func[str(pr[1][0])] >= a and cur[1] == 0 and r2_func[str(cur[0])] >= a:
                            r_score = 0
            '''
            for pr in pair_r_d:
                for cur in neigh_r1:
                    if ((pr[0], cur) in self.conflict_id or (cur, pr[0]) in self.conflict_id):
                       r_score = 0
                        # print(self.G_dataset.r_dict[pr[0][0]], pr[0][1], self.G_dataset.r_dict[cur[0]], cur[1])
                        # print('solve')
                        
                for cur in neigh_r2:
                    if ((pr[1], cur) in self.conflict_id or (cur, pr[1]) in self.conflict_id):
                        
                        r_score = 0
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            
            # score += r_score 
            # score += 1
            # score += float(self.e_sim[p[0]][p[1] - self.split])
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
            if score < 0:
                return  self.get_pair_score_high(e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)

        return pair_node, sigmoid(score)

    def get_high_order_pair_score(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        score = 0
        pair = set()
        for cur in neigh12:
            if cur in cur_link:
                if cur_link[cur] in neigh21:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh21:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            # print(self.search_2_hop_tri(e1, p[0]))
            (t1, t2) = self.search_2_hop_tri(e1, p[0])
            t1 = list(t1)
            t2 = list(t2)
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur1 in t1:
                for cur2 in t2:
                    r1.append((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2)
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                index11 = int(pr[0] / len(t2))
                index12 = pr[0] % len(t2)
                index21 = pr[1]
                cur_score = 1
                cur_score1 = 1
                if t1[index11][0] == e1:
                    cur_score1 = r1_func_r[str(t1[index11][1])]
                else:
                    cur_score1 = r1_func[str(t1[index11][1])]
                if t2[index12][0] == p[0]:
                    cur_score1 *= r1_func_r[str(t2[index12][1])]
                else:
                    cur_score1 *= r1_func[str(t2[index12][1])]
                if tri2[index21][0] == e2:
                    cur_score = min(cur_score1,r2_func_r[str(tri2[index21][1])])       
                else:
                    cur_score = min(cur_score1,r2_func[str(tri2[index21][1])])
                
                r_score += cur_score
            r_score /= len(pair_r)
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
        score /= 2 ** len(pair)
        score1 = 0
        pair = set()
        for cur in neigh11:
            if cur in cur_link:
                if cur_link[cur] in neigh22:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh11:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            (t1, t2) = self.search_2_hop_tri(e2, p[1])
            t1 = list(t1)
            t2 = list(t2)
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur1 in t1:
                for cur2 in t2:
                    r2.append((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2)
            
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                index21 = int(pr[1] / len(t2))
                index22 = pr[1] % len(t2)
                index11 = pr[0]
                cur_score = 1
                cur_score1 = 1
                if tri1[index11][0] == e1:
                    cur_score = min(cur_score,r1_func_r[str(tri1[index11][1])])       
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[index11][1])])
                if t1[index21][0] == e2:
                    cur_score1 = r2_func_r[str(t1[index21][1])]
                else:
                    cur_score1 = r2_func[str(t1[index21][1])]
                if t2[index22][0] == p[1]:
                    cur_score1 *= r2_func_r[str(t2[index22][1])]
                else:
                    cur_score1 *= r2_func[str(t2[index22][1])]
                
                cur_score = min(cur_score, cur_score1)
                r_score += cur_score
            r_score /= len(pair_r)
            score1 += r_score * float(self.e_sim[p[0]][p[1] - self.split])
        score1 /= 2 ** len(pair) 
        return set(), 0.01 * (score + score1) 
        pair = set()
        for cur in neigh12:
            if cur in cur_link:
                if cur_link[cur] in neigh22:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            (t11, t12) = self.search_2_hop_tri(e1, p[0])
            t11 = list(t11)
            t12 = list(t12)
            (t21, t22) = self.search_2_hop_tri(e2, p[1])
            t21 = list(t21)
            t22 = list(t22)
            r1 = []
            r2 = []
            for cur1 in t11:
                for cur2 in t12:
                    r1.append((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2)
            for cur1 in t21:
                for cur2 in t22:
                    r2.append((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2)
            
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                index21 = int(pr[1] / len(t22))
                index22 = pr[1] % len(t22)
                index11 = int(pr[0] / len(t12))
                index12 = pr[0] % len(t12)
                cur_score = 1
                cur_score1 = 1
                cur_score2 = 1
                if t11[index11][0] == e1:
                    cur_score1 = min(cur_score1,r1_func_r[str(t11[index11][1])])       
                else:
                    cur_score1 = min(cur_score1,r1_func[str(t11[index11][1])])
                if t12[index12][0] == p[0]:
                    cur_score1 *= r1_func_r[str(t12[index12][1])]     
                else:
                    cur_score1 *= r1_func[str(t12[index12][1])]
                if t21[index21][0] == e2:
                    cur_score2 = r2_func_r[str(t21[index21][1])]
                else:
                    cur_score2 = r2_func[str(t21[index21][1])]
                if t22[index22][0] == p[1]:
                    cur_score2 *= r2_func_r[str(t22[index22][1])]
                else:
                    cur_score2 *= r2_func[str(t22[index22][1])]
                
                
                cur_score = min(cur_score1, cur_score2)
                r_score += cur_score
            r_score /= len(pair_r)
            score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
        return set(), score

    def get_pair_conflict(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            fact_conflict = 0
            for pr in pair_r:
                direct = 0
                cur_score = 1
                if tri1[pr[0]][0] != e1:
                    direct = 1
                if tri2[pr[1]][0] == e2:
                    if direct == 1:
                        fact_conflict += 1
                else:
                    if direct == 0:
                        fact_conflict += 1
        return fact_conflict

    def get_r_conflict(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return 0
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            fact_conflict = 0
            # print(len(self.conflict_r_pair))
            for pr in pair_r:
                map_r2 = None
                map_r1 = None
                if tri2[pr[1]][1] in self.r_map2:
                    map_r2 = int(self.r_map2[tri2[pr[1]][1]])
                if tri1[pr[0]][1] in self.r_map1:
                    map_r1 = int(self.r_map1[tri1[pr[0]][1]])

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # print(self.G_dataset.r_dict[tri1[pr[0]][1]], self.G_dataset.r_dict[tri2[pr[1]][1]])
                    # print(map_r1, map_r2)
                    # if map_r1:
                        # print(self.G_dataset.r_dict[map_r1])
                    # if map_r2:
                        # print(self.G_dataset.r_dict[map_r2])
                    # print('--------------------------')
                    return 1
        return 0

    def conflict_count(self, cur_model_pair):
        count = 0
        count1 = 0
        conflict_set = defaultdict(set)
        new_model_pair = {}
        for pair in cur_model_pair:
            conflict_set[pair[1]].add(pair[0])
            new_model_pair[pair[0]] = pair[1]
        for ent in conflict_set:
            if len(conflict_set[ent]) > 1:
                # print(self.G_dataset.ent_dict[ent])
                # for cur in self.G_dataset.gid[ent]:
                    # self.read_triple_name(cur)
                # print('************************')
                count += 1
                for e in conflict_set[ent]:
                    new_model_pair[e] = None
                    # print(self.G_dataset.ent_dict[e])
                        
                    # for cur in self.G_dataset.gid[e]:
                        # self.read_triple_name(cur)
                    if self.test_link[str(e)] == str(ent):
                        count1 += 1
            # else:
                # for e in conflict_set[ent]:
                    # if self.test_link[str(e)] != str(ent):
                        # print(str(e) + '\t' + str(ent))
                        # print('in')
                    # print('-----------------------')
        # print(count, count1)
        # exit(0)
        return conflict_set, new_model_pair, count1

    def find_low_confidence1(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        pair_node = set()
        
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                # r1.append(self.r_embed[cur[1]])
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]),dim=0))
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
                # r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            pair_r_d = set() 
            for pr in pair_r:
                direct1 = 0
                direct2 = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                    direct1 = 1
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    direct2 = 1
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add((map_r2, direct2))
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add((map_r1,direct1))
                pair_r_d.add(((tri1[pr[0]][1], direct1), (tri2[pr[1]][1], direct2)))

                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            for pr in pair_r_d:
                for cur in neigh_r1:
                    if ((pr[0], cur) in self.conflict_id or (cur, pr[0]) in self.conflict_id):
                        r_score = 0
                for cur in neigh_r2:
                    if ((pr[1], cur) in self.conflict_id or (cur, pr[1]) in self.conflict_id):
                        r_score = 0
            # score += r_score * float(self.e_sim[p[0]][p[1] - self.split])
            score += r_score 
        return sigmoid(score)
    
    def find_low_confidence(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set(), 0
        else:
            return set(), 1
        
    def get_test_file_mask(self, file, thred, method=''):
        nec_tri = set()
        print(file)
        with open(file, 'r') as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        if self.lang == 'zh':
            with open('../datasets/dbp_z_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_z_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'ja':
            with open('../datasets/dbp_j_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_j_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'y':
            with open('../datasets/D_Y/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/D_Y/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/D_Y/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/D_Y/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'w':
            with open('../datasets/D_W/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/D_W/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/D_W/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/D_W/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'zh_2':
            with open('../datasets/dbp_z_e_wrong/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_z_e_wrong/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_z_e_wrong/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_z_e_wrong/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'ja2':
            with open('../datasets/dbp_j_e_wrong/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_j_e_wrong/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_j_e_wrong/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_j_e_wrong/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'fr2':
            with open('../datasets/dbp_f_e_wrong/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_f_e_wrong/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_f_e_wrong/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_f_e_wrong/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        else:
            with open('../datasets/dbp_f_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_f_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')

    def get_test_file_mask_two(self, file, thred, method=''):
        nec_tri = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                nec_tri.add((int(cur[0]), int(cur[1]), int(cur[2])))
        suff = set()
        for gid1, gid2 in self.test_indices:
            e1 = int(gid1)
            e2 = int(gid2)
            neigh12, neigh11 = self.init_2_hop(e1)
            neigh22, neigh21 = self.init_2_hop(e2)
            for cur in neigh11:
                suff |= self.search_1_hop_tri(e1, cur)
            for cur in neigh12:
                two_hop = self.search_2_hop_tri1(e1, cur)
                for cur1 in two_hop:
                    t1 = cur1[0]
                    t2 = cur1[1]
                    suff |= t1
                    suff |= t2
            for cur in neigh21:
                suff |= self.search_1_hop_tri(e2, cur)
            for cur in neigh22:
                two_hop = self.search_2_hop_tri1(e2, cur)
                for cur1 in two_hop:
                    t1 = cur1[0]
                    t2 = cur1[1]
                    suff |= t1
                    suff |= t2
        self.G_dataset.suff_kgs = suff
        print('sparsity :', 1 - (len(nec_tri) / len(self.G_dataset.suff_kgs)))
        new_kg = self.G_dataset.kgs - nec_tri
        if self.lang == 'zh':
            with open('../datasets/dbp_z_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_z_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_z_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < 19388:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        elif self.lang == 'ja':
            with open('../datasets/dbp_j_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_j_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_j_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
        else:
            with open('../datasets/dbp_f_e/test_triples_1_nec' + method + thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_nec' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
            new_kg = (self.G_dataset.kgs - self.G_dataset.suff_kgs) | nec_tri
            with open('../datasets/dbp_f_e/test_triples_1_suf' + method +thred, 'w') as f1, open('../datasets/dbp_f_e/test_triples_2_suf' + method +thred, 'w') as f2:
                for cur in new_kg:
                    if cur[0] < self.split:
                        f1.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
                    else:
                        f2.write(str(cur[0]) + '\t' + str(cur[1]) + '\t' + str(cur[2]) + '\n')
    def find_relation_conflict(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set()
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        conflict_set = set()
        score = 0
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            for pr in pair_r:
                direct = 0
                cur_score = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                # r_score = max(r_score, cur_score)
                map_r2 = None
                map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add(map_r2)
                # if tri1[pr[0]][1] in self.r_map1:
                map_r1 = self.r_map1[tri1[pr[0]][1]]
                neigh_r2.add(map_r1)


                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                # if ((tri1[pr[0]][1], map_r2) in self.conflict_r_pair) or ((map_r1, tri2[pr[1]][1]) in self.conflict_r_pair) or ((map_r2, tri1[pr[0]][1]) in self.conflict_r_pair) or ((tri2[pr[1]][1], map_r1) in self.conflict_r_pair):
                    # cur_score = 0

                r_score += cur_score
            for pr in pair_r:
                for cur in neigh_r1:
                    if ((tri1[pr[0]][1], cur) in self.conflict_r_pair or (cur, tri1[pr[0]][1]) in self.conflict_r_pair):
                        conflict_set.add((p[0], p[1]))
                for cur in neigh_r2:
                    if ((tri2[pr[1]][1], cur) in self.conflict_r_pair or (cur, tri2[pr[1]][1]) in self.conflict_r_pair):
                        conflict_set.add((p[0], p[1]))
        return conflict_set

    def find_consistency_conflict(self, e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        
        pair = set()
        for cur in neigh1:
            if cur in cur_link:
                if cur_link[cur] in neigh2:
                    pair.add((cur, cur_link[cur]))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        if len(pair) == 0:
            return set()
            # return self.get_high_order_pair_score(e1, e2, r1_func, r1_func_r, r2_func, r2_func_r, cur_link)
        
        # pair = self.explain_bidrect(e1, e2)
        pair_node = set()
        score_list = []
        score = 0
        r_pair = []
        pair = list(pair)
        conflict_set = set()
        for k in range(len(pair)):
            p = pair[k]
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                direct = 0
                cur_score = 1
                cur_score_r = 1
                if tri1[pr[0]][0] == e1:
                    cur_score = min(cur_score, r1_func_r[str(tri1[pr[0]][1])])
                    r_1 = (tri1[pr[0]][1], 0)
                else:
                    cur_score = min(cur_score,r1_func[str(tri1[pr[0]][1])])

                    r_1 = (tri1[pr[0]][1], 1)
                if tri2[pr[1]][0] == e2:
                    cur_score = min(cur_score,r2_func_r[str(tri2[pr[1]][1])])       
                    r_2 = (tri2[pr[1]][1], 0)
                else:
                    cur_score = min(cur_score,r2_func[str(tri2[pr[1]][1])])
                    r_2 = (tri2[pr[1]][1], 1)
                
                # map_r2 = None
                # map_r1 = None
                # if tri2[pr[1]][1] in self.r_map2:
                    # map_r2 = int(self.r_map2[tri2[pr[1]][1]])
                
                map_r2 = self.r_map2[tri2[pr[1]][1]]
                neigh_r1.add(map_r2)
                # if tri1[pr[0]][1] in self.r_map1:
                    # print(self.r_map1[tri1[pr[0]][1]])
                    # map_r1 = int(self.r_map1[tri1[pr[0]][1]])
                map_r1 = self.r_map1[tri1[pr[0]][1]]

                neigh_r2.add(map_r1)
                # print(tri1[pr[0]][1], self.r_map2[tri2[pr[1]][1]])
                r_score += cur_score
                cur_r_pair.append((r_1, r_2))
            r_pair.append(cur_r_pair)
            score_list.append(r_score)
            pair_node.add((str(p[0]) +'\t'+ str(p[1]), r_score))
        
        
        
        for i in range(len(r_pair) - 1):
            for j in range(i + 1, len(r_pair)):
                for cur1 in r_pair[i]:
                    for cur2 in r_pair[j]:
                        cur_sim = self.e_sim[pair[i][0]][pair[j][1] - self.split]
                        # print(cur1[0][1], cur2[1][1], self.r_map1[cur1[0][0]], cur1[0][0], cur2[1][0], self.r_map2[cur2[1][0]])
                        map1 = self.r_map2[cur2[1][0]]
                        map2 = self.r_map1[cur1[0][0]]
                        # print(self.line_id)
                        # exit(0)
                        # print((cur1[0][0], cur1[0][1], map1, cur2[1][1]))
                        if (((cur1[0][0], cur1[0][1]), (map1, cur2[1][1]))) in self.line_id or (((map2, cur1[0][1]), (cur2[1][0], cur2[1][1]))) in self.line_id:
                            conflict_set.add((e1, e2))
                            # print(self.G_dataset.r_dict[cur1[0][0]], cur1[0][1], self.G_dataset.r_dict[cur2[1][0]], cur2[1][1])
                            # print(self.G_dataset.ent_dict[e1] + '\t' + self.G_dataset.ent_dict[e2], self.test_link[str(e1)] ==  str(e2))
                            
                                
                            # print(self.G_dataset.ent_dict[pair[i][0]] + '\t' + self.G_dataset.ent_dict[pair[j][1]])
                            # if str(pair[i][0]) in self.test_link:
                                # print(self.test_link[str(pair[i][0])] ==  str(pair[j][1]))
                        '''
                        if cur1[0][1] == cur2[1][1] and (self.r_map1[cur1[0][0]] == cur2[1][0] or cur1[0][0] == self.r_map2[cur2[1][0]]):
                            # print('exist same relation')
                            # print(self.G_dataset.r_dict[cur1[0][0]], self.G_dataset.r_dict[cur1[1][0]])
                            if (cur1[0][1] == 0 and min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]) >= 0.8):
                                # print(self.G_dataset.r_dict[cur1[0][0]], self.G_dataset.r_dict[cur2[1][0]])
                                # conflict_set.add((e1, e2))
                                conflict_set.add((pair[i][0], pair[i][1]))
                                conflict_set.add((pair[j][0], pair[j][1]))
                                # print('exist same relation and r_func high')
                                # if str(pair[i][0]) in self.test_link:
                                    # print(self.e_sim[pair[i][0]][pair[i][1] - self.split], self.test_link[str(pair[i][0])] ==  str(pair[i][1]))
                                    # print(self.G_dataset.ent_dict[e1] + '\t' + self.G_dataset.ent_dict[e2])
                                    # print(self.G_dataset.ent_dict[pair[i][0]] + '\t' + self.G_dataset.ent_dict[pair[i][1]])
                                # score_list[i] *= min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])])
                                # score_list[j] *= min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]) 
                                # score_list[i] *= (min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # score_list[j] *= (min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # score_list[i] = 0
                                # score_list[j] = 0
                                # self.e_sim[e1][e2 - self.split] *= (0.5 + 1 - min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]))
                                # score_list.append(min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # print(self.e_sim[pair[i][0]][pair[j][1] - self.split])
                                # pair.append((pair[i][0], pair[j][1]))
                                # self.e_sim[pair[i][0]][pair[j][1] - self.split] = max(min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]), cur_sim)
                            elif (cur1[0][1] == 1 and min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]) >= 0.8):
                                # print(self.G_dataset.r_dict[cur1[0][0]], self.G_dataset.r_dict[cur2[1][0]])
                                # conflict_set.add((e1,e2))
                                conflict_set.add((pair[i][0], pair[i][1]))
                                conflict_set.add((pair[j][0], pair[j][1]))
                                # if str(pair[i][0]) in self.test_link:
                                # if str(pair[i][0]) in self.test_link:
                                    # print(self.e_sim[pair[i][0]][pair[i][1] - self.split], self.test_link[str(pair[i][0])] ==  str(pair[i][1]))
                                    # print(self.e_sim[pair[i][0]][pair[j][1] - self.split], self.test_link[str(pair[i][0])] ==  str(pair[j][1]))
                                # print('exist same relation and r_func high')
                                    # print(self.G_dataset.ent_dict[e1] + '\t' + self.G_dataset.ent_dict[e2])
                                    # print(self.G_dataset.ent_dict[pair[i][0]] + '\t' + self.G_dataset.ent_dict[pair[i][1]])
                                # self.e_sim[pair[i][0]][pair[j][1] - self.split] = max(min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]), cur_sim)
                                # score_list[i] *= min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])])
                                # self.e_sim[e1][e2 - self.split] *= (0.5 + 1 - min(r1_func_r[str(cur1[0][0])], r2_func_r[str(cur2[1][0])]))
                                # score_list[j] *= min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])])
                                # score_list[i] = 0
                                # score_list[j] = 0
                                # print(self.e_sim[pair[i][0]][pair[j][1] - self.split])
                                # score_list.append(min(r1_func[str(cur1[0][0])], r2_func[str(cur2[1][0])]))
                                # pair.append((pair[i][0], pair[j][1]))
                        '''
        return conflict_set
    
    def find_common_neigh(self, neigh1, neigh2, cur_link):
        pair = set()
        for cur in neigh1:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh2:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))
        return pair
    def find_could_not_do(self):
        cur_link = self.test_link
        count = 0
        for i in range(len(self.test_pair)):
            gid1, gid2 = self.test_pair[i]
            e1 = int(gid1)
            e2 = int(gid2)
            neigh12, neigh11 = self.init_2_hop(e1)
            neigh22, neigh21 = self.init_2_hop(e2)
            pair1 = self.find_common_neigh(neigh11, neigh21, cur_link)
            for e3 in self.G_dataset.entity2:
                if e3 != e2:
                    neigh3 = self.get_1_hop(e3)
                    pair2 = self.find_common_neigh(neigh11, neigh3, cur_link)
                    if (len(pair1 & pair2) == len(pair1) and len(pair1 & pair2) == len(pair2)) and len(pair1) > 0 and neigh3 == neigh21:
                        count += 1
                        print(self.G_dataset.ent_dict[e1], self.G_dataset.ent_dict[e2], self.G_dataset.ent_dict[e3])
                        break
                        '''
                        neigh32, neigh31 = self.init_2_hop(e3)
                        pair3 = self.find_common_neigh(neigh11 | neigh12, neigh21 | neigh22, cur_link)
                        pair4 = self.find_common_neigh(neigh11 | neigh12, neigh31 | neigh32, cur_link)
                        if (len(pair3 & pair4) == len(pair3) and len(pair3 & pair4) == len(pair3)):
                            count += 1
                            print(self.G_dataset.ent_dict[e1], self.G_dataset.ent_dict[e2], self.G_dataset.ent_dict[e3])
                            break
                        '''

        print(count / len(cur_link))

    def get_1_hop(self, e):
        neigh1 = set()
        for cur in self.G_dataset.gid[e]:
            if cur[0] != int(e):
                neigh1.add(int(cur[0]))
            else:
                neigh1.add(int(cur[2]))
        return neigh1

    def init_2_hop(self, e1):
        neigh2 = set()
        neigh1 = self.get_1_hop(e1)
        for ent in neigh1:
            neigh2 |= self.get_1_hop(ent)
        neigh2.remove(e1)

        return neigh2 - neigh1 , neigh1

    def search_2_hop_tri(self, e, tar):
        neigh1 = self.get_1_hop(e)
        tri2 = []
        cur1 = set()
        cur2 = set()
        for ent in neigh1:
            neigh2 = self.get_1_hop(ent)
            if tar in neigh2:
                t1 = self.search_1_hop_tri(e, ent)
                t2 = self.search_1_hop_tri(ent, tar)
                tri2.append((t1, t2))
                cur1 |= t1
                cur2 |= t2
        # print(cur1, cur2)
        return cur1, cur2

    def search_2_hop_tri1(self, e, tar):
        neigh1 = self.get_1_hop(e)
        tri2 = []
        cur1 = set()
        cur2 = set()
        for ent in neigh1:
            neigh2 = self.get_1_hop(ent)
            if tar in neigh2:
                t1 = self.search_1_hop_tri(e, ent)
                t2 = self.search_1_hop_tri(ent, tar)
                tri2.append((t1, t2, ent))
                cur1 |= t1
                cur2 |= t2
        # print(cur1, cur2)
        return tri2

    def read_triple_name(self, triple):
        print(self.G_dataset.ent_dict[triple[0]], self.G_dataset.r_dict[triple[1]],self.G_dataset.ent_dict[triple[2]])
    def get_r_func(self):
        if self.lang == 'de':
            tri1 = read_tri('/data/xbtian/Explain/MTransE/datasets/OpenEA/EN_DE_15K/triples_1')
            tri2 = read_tri('/data/xbtian/Explain/MTransE/datasets/OpenEA/EN_DE_15K/triples_2')
            r, _ = read_link('/data/xbtian/Explain/MTransE/datasets/OpenEA/EN_DE_15K/rel_dict')
        elif self.lang == 'y':
            tri1 = read_tri('/data/xbtian/Explain/MTransE/datasets/OpenEA/D_Y_15K/triples_1')
            tri2 = read_tri('/data/xbtian/Explain/MTransE/datasets/OpenEA/D_Y_15K/triples_2')
            r, _ = read_link('/data/xbtian/Explain/MTransE/datasets/OpenEA/D_Y_15K/rel_dict')
        elif self.lang == 'w' or self.lang == 'w2':
            tri1 = read_tri('/data/xbtian/Explain/MTransE/datasets/OpenEA/D_W_15K/triples_1')
            tri2 = read_tri('/data/xbtian/Explain/MTransE/datasets/OpenEA/D_W_15K/triples_2')
            r, _ = read_link('/data/xbtian/Explain/MTransE/datasets/OpenEA/D_W_15K/rel_dict')
        elif self.lang == 'zh2':
            tri1 = read_tri('../datasets/z_e_2/triples_1')
            tri2 = read_tri('../datasets/z_e_2/triples_2')
            r, _ = read_link('../datasets/z_e_2/rel_dict')
        elif self.lang == 'zh_2':
            tri1 = read_tri('../datasets/dbp_z_e_wrong/triples_1')
            tri2 = read_tri('../datasets/dbp_z_e_wrong/triples_2')
            r, _ = read_link('../datasets/dbp_z_e_wrong/rel_dict')
        elif self.lang == 'ja2':
            tri1 = read_tri('../datasets/dbp_j_e_wrong/triples_1')
            tri2 = read_tri('../datasets/dbp_j_e_wrong/triples_2')
            r, _ = read_link('../datasets/dbp_j_e_wrong/rel_dict')
        elif self.lang == 'fr2':
            tri1 = read_tri('../datasets/dbp_f_e_wrong/triples_1')
            tri2 = read_tri('../datasets/dbp_f_e_wrong/triples_2')
            r, _ = read_link('../datasets/dbp_f_e_wrong/rel_dict')
        else:
            tri1 = read_tri('/data/xbtian/ContEA-explain/datasets/' + self.lang + '-en_f/base/triples_1')
            tri2 = read_tri('/data/xbtian/ContEA-explain/datasets/' + self.lang + '-en_f/base/triples_2')
            r, _ = read_link('/data/xbtian/ContEA-explain/datasets/' + self.lang + '-en_f/base/rel_dict')
        r1 = defaultdict(set)
        r2 = defaultdict(set)
        r1_func = defaultdict(int)
        r2_func = defaultdict(int)
        r1_func_r = defaultdict(int)
        r2_func_r = defaultdict(int)
        for cur in tri1:
            r1[cur[1]].add((cur[0], cur[2]))
        for cur in tri2:
            r2[cur[1]].add((cur[0], cur[2]))
        
        for cur in r1:
            x = defaultdict(int)
            for t in r1[cur]:
                x[t[0]] = 1
            r1_func[cur] = len(x) / len(r1[cur])
            x_r = defaultdict(int)
            for t in r1[cur]:
                x_r[t[1]] = 1
            r1_func_r[cur] = len(x_r) / len(r1[cur])
            
        
        for cur in r2:
            x = defaultdict(int)
            for t in r2[cur]:
                x[t[0]] = 1
            r2_func[cur] = len(x) / len(r2[cur])
            x_r = defaultdict(int)
            for t in r2[cur]:
                x_r[t[1]] = 1
            r2_func_r[cur] = len(x_r) / len(r2[cur])
        '''
        for cur in r1_func:
            # print(self.G_dataset.r_dict[int(cur)], r1_func[cur], r1_func_r[cur])
            print(str(cur) + '\t'+ str(r1_func[cur]) + '\t' + str(r1_func_r[cur]))
        for cur in r2_func:
            print(str(cur) + '\t'+ str(r2_func[cur]) + '\t' + str(r2_func_r[cur]))
            # print(self.G_dataset.r_dict[int(cur)], r2_func[cur], r2_func_r[cur])
        exit(0)
        '''
        '''
        for cur in r1_func:
            if r1_func[cur] == 1:
                print(cur)
        for cur in r1_func_r:
            if r1_func_r[cur] == 1:
                print(cur)
        for cur in r2_func:
            if r2_func[cur] == 1:
                print(cur)
        for cur in r2_func_r:
            if r2_func_r[cur] == 1:
                print(cur)
        exit(0)
        '''
        return r1_func, r1_func_r, r2_func, r2_func_r
    def explain_ours5(self, e1, e2):
        exp_tri1 = []
        exp_tri2 = []
        cur_link = self.model_link
        neigh12, neigh11 = self.init_2_hop(e1)
        neigh22, neigh21 = self.init_2_hop(e2)
        score = 0
        pair = set()
        for cur in neigh12:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh21:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh21:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            # print(self.search_2_hop_tri(e1, p[0]))
            two_hop_list = self.search_2_hop_tri1(e1, p[0])
            
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            index = 0
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r1.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e1]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            for cur in tri2:
                r2.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e2]), dim=0))
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)

            for i in range(len(pair_r)):
                pr = pair_r[i]
                exp_tri1 += [two_hop[pr[0]][0]]
                exp_tri1 += [two_hop[pr[0]][1]]
                exp_tri2 += [tri2[pr[1]]]
                
        pair = set()
        for cur in neigh11:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh22:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh11:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        score_list = []
        r_pair = []
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            two_hop_list = self.search_2_hop_tri1(e2, p[1])
            
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(torch.cat((self.r_embed[cur[1]], self.e_embed[e1]), dim=0))
            two_hop = []
            for cur in two_hop_list:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r2.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e2]) / 2), dim=0))
                        two_hop += [(cur1, cur2)]
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            r_score = 0
            neigh_r1 = set()
            neigh_r2 = set()
            judge = 0
            cur_r_pair = []

            for i in range(len(pair_r)):
                pr = pair_r[i]
                exp_tri1 += [tri1[pr[0]]]
                exp_tri2 += [two_hop[pr[1]][0]]
                exp_tri2 += [two_hop[pr[1]][1]]
                
        pair = set()
        for cur in neigh12:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh22:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh12:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh22:
                    pair.add((cur, int(self.train_link[str(cur)])))
        pair = list(pair)
        for k in range(len(pair)):
            p = pair[k]
            two_hop_list1 = self.search_2_hop_tri1(e1, p[0])

            two_hop_list2 = self.search_2_hop_tri1(e2, p[1])

            r1 = []
            r2 = []
            two_hop1 = []
            for cur in two_hop_list1:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r1.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e1]) / 2), dim=0))
                        two_hop1 += [(cur1, cur2)]
            
            two_hop2 = []
            for cur in two_hop_list2:
                e3 = cur[2]
                t1 = cur[0]
                t2 = cur[1]
                t1 = list(t1)
                t2 = list(t2)
                for cur1 in t1:
                    for cur2 in t2:
                        r2.append(torch.cat(((self.r_embed[cur1[1]] + self.r_embed[cur2[1]]) / 2, (self.e_embed[e3] + self.e_embed[e2]) / 2), dim=0))
                        two_hop2 += [(cur1, cur2)]
            # print(two_hop2)
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)

            for i in range(len(pair_r)):
                pr = pair_r[i]
                # print(two_hop1[pr[0]][0])
                exp_tri1 += [two_hop1[pr[0]][0]]
                exp_tri1 += [two_hop1[pr[0]][1]]
                exp_tri2 += [two_hop2[pr[1]][0]]
                exp_tri2 += [two_hop2[pr[1]][1]]
        # print(exp_tri1, exp_tri2)
        return exp_tri1, exp_tri2
    

    
    
    def explain_ours4(self, e1, e2):
        neigh1, neigh2 = self.init_1_hop(e1, e2)
        
        cur_link = self.model_link
        pair = set()
        for cur in neigh1:
            if str(cur) in cur_link:
                if int(cur_link[str(cur)]) in neigh2:
                    pair.add((cur, int(cur_link[str(cur)])))
        for cur in neigh1:
            if str(cur) in self.train_link:
                if int(self.train_link[str(cur)]) in neigh2:
                    pair.add((cur, int(self.train_link[str(cur)])))


        # pair = self.explain_bidrect(e1, e2)[:5]
        tri1_list = []
        tri2_list = []
        # print(pair)
        for p in pair:
            tri1 = list(self.search_1_hop_tri(e1, p[0]))
            tri2 = list(self.search_1_hop_tri(e2, p[1]))
            r1 = []
            r2 = []
            for cur in tri1:
                r1.append(self.r_embed[cur[1]])
            for cur in tri2:
                r2.append(self.r_embed[cur[1]])
        
            r1 = torch.stack(r1)
            r2 = torch.stack(r2)
            sim = torch.mm(r1, r2.t())
            pre1 = (-sim).argsort()
            pre2 = (-(sim.t())).argsort()
            pair_r = self.bidirect_match(pre1, pre2)
            new_tri1 = []
            new_tri2 = []
            
            for pr in pair_r:
                new_tri1.append(tri1[pr[0]])
                new_tri2.append(tri2[pr[1]])

            tri1_list += new_tri1
            tri2_list += new_tri2
        return tri1_list, tri2_list, pair

    def extract_rule2(self, file, file_out):
        local_rule = {}
        train = {}
        test = {}
        ground = {}
        align = {}
        _, r_dict = read_link('/data/xbtian/ContEA-explain/datasets/zh-en_f/base/rel_dict')
        for i in range(len(self.G_dataset.test_pair)):
            ground[self.G_dataset.ent_dict[self.G_dataset.test_pair[i][0]]] = self.G_dataset.ent_dict[self.G_dataset.test_pair[i][1]]

        with open(file) as f:
            lines = f.readlines()
            align_entities = ''
            rule = []
            ent = {}
            r = {}
            local_rule = {}
            begin = 1
            for i in trange(len(lines)):
                cur = lines[i].strip().split('\t')
                # print(len(cur))
                if len(cur) == 1 and cur[0][0] == '*':
                    
                    head = set()
                    for cur in rule:
                        # term = 'r' + r_dict[cur[1]]
                        term = cur[1].split('/')[-1]
                        head.add((ent[cur[0]],  term , ent[cur[2]]))
                    local_rule[align_entities] = ((head, body))
                    rule = []
                    begin = 1
                elif len(cur) == 1 and cur[0][0] == '-':
                    continue
                elif len(cur) == 2 and begin == 1:
                    begin = 0
                    ent = {}
                    r = {}
                    count = 0
                    v = 'A'
                    '''
                    judge = 1
                    if ground[cur[0]] != cur[1]:
                        judge = 0
                    
                    
                    align_entities = cur[0] + '\t' + cur[1] + '\t' + str(judge)
                    '''
                    align_entities = cur[0] + '\t' + cur[1] + '\t' + str(float(self.e_sim[int(self.G_dataset.id_ent[cur[0]]), int(self.G_dataset.id_ent[cur[1]])- self.split] ))
                    ent[cur[0]] = chr(ord(v) + count)
                    count += 1 
                    ent[cur[1]] = chr(ord(v) + count)
                    count += 1
                    e1 = cur[0]
                    e2 = cur[1]
                    
                    body = [(ent[cur[0]], 'sameAs', ent[cur[1]])]
                elif len(cur) == 2 and begin == 0:
                    ent[cur[0]] = chr(ord(v) + count)
                    count += 1
                    ent[cur[1]] = chr(ord(v) + count)
                    count += 1
                    rule.append((cur[0], '/sameAs', cur[1]))
                elif len(cur) == 3:
                    rule.append((cur[0], cur[1], cur[2]))
        with open(file_out, 'w') as f:
            for cur in local_rule:
                head, body = local_rule[cur]
                head = list(head)
                cur = cur.split('\t')
                
                '''
                for i in range(len(head)):
                    if i != len(head) - 1:
                        f.write(head[i][1] + '(' + head[i][0] + ',' + head[i][2] + ')' + ' & ')
                    else:
                        f.write(head[i][1] + '(' + head[i][0] + ',' + head[i][2] + ')\n')
                '''
                for i in range(len(head)):
                    if i != len(head) - 1:
                        f.write(head[i][1] + '(' + head[i][0] + ',' + head[i][2] + ')' + ' & ')
                    else:
                        f.write(head[i][1] + '(' + head[i][0] + ',' + head[i][2] + ') => ')
                f.write(body[0][1] + '(' + body[0][0] + ',' + body[0][2] + ')' + cur[2] + '\n')

    def pattern_process(self, e, l=2):
        p = []
        if l == 1:
            p_embed = torch.zeros(len(self.G_dataset.gid[e]), self.embed.shape[1] + self.r_embed.shape[1]) 
        else:
            p_embed = torch.zeros(len(self.G_dataset.pattern[e]), self.embed.shape[1] + self.r_embed.shape[1]) 
        i = 0
        if l == 2:
            for cur in self.G_dataset.pattern[e]:
                p.append(cur)

                if len(cur) == 3:
                    if cur[0] == e1:
                        p_embed[i] = torch.cat((self.embed[cur[2]], self.r_embed[cur[1] + 1]), dim=0)
                    else:
                        p_embed[i] = torch.cat((self.embed[cur[0]], self.r_embed[cur[1] + 1]), dim=0)
                else:
                    if cur[0][0] == e1:
                        if cur[0][2] == cur[1][0]:
                            p_embed[i] = torch.cat(((self.embed[cur[0][2]] + self.embed[cur[1][2]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1] + 1]) / 2), dim=0)
                        else:
                            p_embed[i] = torch.cat(((self.embed[cur[0][2]] + self.embed[cur[1][0]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1]] + 1) / 2), dim=0)
                    else:
                        if cur[0][0] == cur[1][0]:
                            p_embed[i] = torch.cat(((self.embed[cur[0][0]] + self.embed[cur[1][2]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1] + 1]) / 2), dim=0)
                        else:
                            p_embed[i] = torch.cat(((self.embed[cur[0][0]] + self.embed[cur[1][0]]) / 2, (self.r_embed[cur[0][1] + 1] +self.r_embed[cur[1][1] + 1]) / 2), dim=0)
                i += 1
        else:
            for cur in self.G_dataset.gid[e]:
                p.append(cur)
                if cur[0] == e1:
                    p_embed[i] = torch.cat((self.embed[cur[2]], self.r_embed[cur[1]]), dim=0)
                else:
                    p_embed[i] = torch.cat((self.embed[cur[0]], self.r_embed[cur[1]]), dim=0)

                i += 1
        return p, p_embed

    def extract_subgraph(self, e1, e2, l=1):
        if l == 1:
            tri = self.G_dataset.gid[e1] + self.G_dataset.gid[e2]
        # print(tri)
        e_dict = {}
        r_dict = {}
        i = 0
        j = 0
        l = 0
        kg1_index = set()
        kg2_index = set()
        for cur in tri:
            if cur[0] not in e_dict:
                e_dict[cur[0]] = i
                if l >= len(self.G_dataset.gid[e1]):
                    kg2_index.add(i)
                else:
                    kg1_index.add(i)
                i += 1
            if cur[2] not in e_dict:
                e_dict[cur[2]] = i
                if l >= len(self.G_dataset.gid[e1]):
                    kg2_index.add(i)
                else:
                    kg1_index.add(i)
                i += 1
            if cur[1] not in r_dict:
                r_dict[cur[1]] = j
                j += 1
            
            l += 1
        new_tri = []
        '''
        for cur in tri:
            new_tri.add((e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]))
            new_tri.add((e_dict[cur[2]], r_dict[cur[1]] + len(r_dict), e_dict[cur[0]]))
        '''
        tri1 = []
        tri2 = []
        for cur in tri:
            if e_dict[cur[0]] in kg1_index:
                tri1.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
            else:
                tri2.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
            new_tri.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
        for cur in tri:
            new_tri.append([e_dict[cur[2]], r_dict[cur[1]] + len(r_dict), e_dict[cur[0]]])

        e_dict_r = {}
        r_dict_r = {}
        for cur in e_dict:
            # print(cur, e_dict[cur], self.G_dataset.ent_dict[cur])
            e_dict_r[e_dict[cur]] = cur
        for cur in r_dict:
            # print(cur, r_dict[cur], self.G_dataset.r_dict[cur])
            r_dict_r[r_dict[cur]] = cur
        return torch.Tensor(new_tri).long(), e_dict, r_dict, list(kg1_index), list(kg2_index), tri1, tri2, e_dict_r, r_dict_r

    def Trans_Process(self, e):
        i = 0
        p_embed = torch.zeros(len(self.G_dataset.gid[e]), self.embed.shape[1]) 
        for cur in self.G_dataset.gid[e]:
            if cur[0] == e:
                p_embed[i] = self.embed[cur[2]] +  self.r_embed[cur[1]]
            else:
                p_embed[i] = self.embed[cur[0]] -  self.r_embed[cur[1]]
            i += 1
        return p_embed

    def bidirect_match(self, neigh_pre1, neigh_pre2, neigh_list1=None, neigh_list2=None, sim=None):
        res = []
        for i in range(neigh_pre1.shape[0]):
            select = neigh_pre1[i][0]
            if i == neigh_pre2[select][0]:
                # res.append([[i, select], sim[i][select]])
                res.append((i, int(select)))
        # res.sort(key=lambda x:x[1], reverse=True)
        # match = []
        # for cur in res:
            # match.append(cur[0])
        return res

    def greedy_match(self, neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, sim):
        res = []
        for i in range(neigh_pre1.shape[0]):
            select = neigh_pre1[i][0]
            res.append((i, int(select)))
        # res.sort(key=lambda x:x[1], reverse=True)
        # match = []
        # for cur in res:
            # match.append(cur[0])
        return res

    def get_proxy_model(self, e1, e2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index = self.extract_subgraph(e1, e2, 1)
        model = CompGCNLayer(self.embed.shape[1], self.r_embed.shape[1], len(r_dict))
        ent_embed = torch.zeros(len(e_dict), self.embed.shape[1]).cuda()
        r_embed = torch.zeros(2 * len(r_dict), self.r_embed.shape[1]).cuda()
        for cur in e_dict:
            ent_embed[e_dict[cur]] = self.embed[cur].cuda()
        for cur in r_dict:
            r_embed[r_dict[cur]] = self.r_embed[cur + 1].cuda()
            r_embed[r_dict[cur] + len(r_dict)] = self.r_embed[cur + int(self.r_embed.shape[0] / 2) + 1].cuda()
        pre_sim = F.cosine_similarity(ent_embed[e_dict[e1]], ent_embed[e_dict[e2]], dim=0)
        y = torch.mm(ent_embed[kg1_index], ent_embed[kg2_index].t())
        ent_embed[e_dict[e1]] = torch.zeros(self.embed.shape[1]).cuda()
        ent_embed[e_dict[e2]] = torch.zeros(self.embed.shape[1]).cuda()
        # print(ent_embed)
        # print(r_embed)
        # print(ent_embed)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
        # print(explainer.triple_mask.requires_grad)
        model.train()
        # print(kg1_index, kg2_index)
        pre_sim = 0
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(1000):
            optimizer.zero_grad()
            # print(ent_embed)
            h, r = model(ent_embed, r_embed, new_graph)
            # print(loss)
            h = h / (torch.linalg.norm(h, dim=-1, keepdim=True) + 1e-5)
            # x = torch.mm(h[kg1_index], h[kg2_index].t())
            sim1 = torch.mm(h[e_dict[e1]].unsqueeze(0), h[kg2_index].t())
            label1 = torch.Tensor([int(e_dict[e2])]).cuda()
            sim2 = torch.mm(h[e_dict[e2]].unsqueeze(0), h[kg1_index].t())
            label2 = torch.Tensor([int(e_dict[e1])]).cuda()
            loss = criterion(sim1, label1.long()) + criterion(sim2, label2.long())
            # kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
            # print(x.softmax(dim=-1).log(), y.softmax(dim=-1))
            # print(x, y, kl)
            # exit(0)
            # loss = F.pairwise_distance(h[e_dict[e1]], h[e_dict[e2]], p=2) 
            # loss = kl
            sim = F.cosine_similarity(h[e_dict[e1]], h[e_dict[e2]], dim=0)
            print('sim:',sim)
            if sim > pre_sim:
                pre_sim = sim
            else:
                break                
            print(loss)
            loss.backward()
            
            optimizer.step()
        return model, ent_embed, r_embed, e_dict, r_dict, new_graph

    def get_proxy_model_ori(self, e1, e2):
        new_graph, e_dict, r_dict, kg1_index, kg2_index, tri1, tri2, e_dict_r, r_dict_r = self.extract_subgraph(e1, e2, 1)
        
        e_embed = torch.zeros(len(e_dict), self.e_embed.shape[1])
        r_embed = torch.zeros(len(r_dict), self.r_embed.shape[1])
        for cur in e_dict:
            e_embed[e_dict[cur]] = self.e_embed[cur]
        # r_embed[0] = self.r_embed[0].cuda()
        # r_embed[len(r_dict)] = self.r_embed[int(self.r_embed.shape[0] / 2)].cuda()
        for cur in r_dict:
            r_embed[r_dict[cur]] = self.r_embed[cur]
           


        p_embed1 = self.Trans_Process(e1)
        p_embed2 = self.Trans_Process(e2)
        model = Proxy(p_embed1, p_embed2)
        
        return model,  e_dict, r_dict, e_dict_r, r_dict_r, new_graph


    def model_score(self, model, e1, e2, tri1, tri2):
        ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, tri1, tri2, True)
        me1, me2 = model.get_embeddings([e1], [e2], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)

        return me1, me2

    def change_pattern_id(self, p, e_dict, r_dict):
        new_p = []
        for cur in p:
            if len(cur) == 3:
                new_p.append([e_dict[cur[0]], r_dict[cur[1]], e_dict[cur[2]]])
        return new_p

    def extract_feature(self, p_embed1, p_embed2, p1, p2):
        neigh_sim = torch.mm(p_embed1, p_embed2.t())
        _, index = (-neigh_sim).sort()
        select_index = index[:, : 3]
        p = []
        for i in range(len(p1)):
            for j in range(select_index[i].shape[0]):
                p.append([p1[i], p2[select_index[i][j]]])
        return p, torch.zeros(len(p))
    
    
    def std_filter(self, data, thred):
        mean = np.mean(data)  # 计算数据均值
        std = np.std(data)    # 计算数据标准差
        
        return mean, std


    
            

            
    def semantic_match_soft(self, p_embed1, p_embed2):
        global_sim = torch.mm(p_embed1, p_embed2.t()).reshape(1, -1)
        value, index = torch.sort(global_sim, descending=True)
        return index[0]



    def semantic_match(self, p_embed1, p_embed2, p1, p2, model, e_dict, r_dict, e1, e2, base_model='Dual_AMN'):
        global_sim = torch.mm(p_embed1, p_embed2.t())
        neigh_pre1 = (-global_sim).argsort()
        neigh_pre2 = (-(global_sim.t())).argsort()
        time1 = time.time()
        # global_res = self.max_weight_match(neigh_pre1, neigh_pre2, list(range(len(p1))), list(range(len(p1), len(p1) + len(p2))), global_sim, 0)
        global_res = self.bidirect_match(neigh_pre1, neigh_pre2, list(range(len(p1))), list(range(len(p1), len(p1) + len(p2))), global_sim)
        time2 = time.time()
        print('max_match time :', time2 - time1)

        pair = []
        sim_list = []

        for cur in global_res:
            if cur[0] < len(p1):
                sim = global_sim[cur[0], cur[1] ]
                pair.append([(cur[0], cur[1] ), sim])
            else:
                sim = global_sim[cur[1], cur[0]]
                pair.append([(cur[1], cur[0] ), sim])
            sim_list.append(sim)
        mean, std = self.std_filter(sim_list, 1)
        pair.sort(key=lambda x: x[1], reverse=True)
        # return pair
        
        proxy_e1, proxy_e2 = model.all_aggregation()
        all_suf = F.cosine_similarity(proxy_e1, proxy_e2, dim=0)


        print(all_suf)
        filter_pair = []
        select_kg1 = set()
        select_kg2 = set()
        exp_pair = []
        all_pair = []
        # print(pair)
        for cur in pair:
            print(cur[1], mean-std, std)
            if float(cur[1]) <= (mean - std - 0.01):
                print('jump')
            else:
                cur = cur[0]
                all_pair.append([cur[0], cur[1]])
                tri1 = [cur[0]]
                tri2 = [cur[1]]
                proxy_e1, proxy_e2 = model.aggregation(tri1, tri2)
                v_suf = F.cosine_similarity(proxy_e1, proxy_e2, dim=0)
                
                print(v_suf)
                if v_suf >= all_suf - 0.2:
                    filter_pair.append([cur[0], cur[1]])
                    select_kg1.add(int(cur[0]))
                    select_kg2.add(int(cur[1]))
                    exp_pair.append([int(cur[0]), int(cur[1])])
                # exp_pair.append([cur[0], cur[1]])
            
            
        print(len(exp_pair) / len(pair))
        if len(exp_pair) / len(pair) == 0:
            exp_pair = all_pair
        return filter_pair, exp_pair, list(select_kg1), list(select_kg2),list(set(list(range(len(p1)))) - select_kg1), list(set(list(range(len(p2)))) - select_kg2)
        
    def judge(self, e1, e2, model, tri1, tri2, no_tri1, no_tri2, e_dict, r_dict):
        retain = self.split
        dec = 0
        try:
            if len(tri1) and len(tri2):
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, tri1, tri2, True, len(e_dict), len(r_dict))
                proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                _, suf_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
                retain = torch.where(suf_rank[0] == e1)[0][0]
                print('suf rank:', retain)
            else:
                retain = self.split

            if len(no_tri1) and len(no_tri2):
                ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size = self.G_dataset.reconstruct_search(e1, e2, no_tri1, no_tri2, True, len(e_dict), len(r_dict))
                proxy_e1, proxy_e2 = model.get_embeddings([e_dict[e1]], [e_dict[e2]], ent_adj, rel_adj, node_size, rel_size, adj_list, r_index, r_val, triple_size, None)
                _, nec_rank = torch.mm(proxy_e1.cuda(), self.embed[self.split:].t()).sort(descending=True)
                # print(nec_rank.shape)
                dec = torch.where(nec_rank[0] == e1)[0][0]
                print('nec rank:', dec)
            else:
                dec = self.split
            
            if retain < 2 and dec > 0:
                return True
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print('WARNING: out of memory')
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                else:
                    raise exception

    def explain_ours(self, e1, e2):
        # model, ent_embed, r_embed, e_dict, r_dict, graph = self.get_proxy_model_ori(e1, e2)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        # exit(0)
        soft_match = 1
        # print(graph.shape[0])
        mask = []
        Y = []
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        # print(p1, p2)
        # print(graph)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        candidate_pair, exp_pair, kg1, kg2, f_kg1, f_kg2 = self.semantic_match(p_embed1, p_embed2, new_p1, new_p2, model, e_dict, r_dict, e1, e2)
        # print(p)
        tri1 = []
        tri2 = []
        exp_tri1 = []
        exp_tri2 = []
        no_tri1 = list(range(len(new_p1)))
        no_tri2 = list(range(len(new_p2)))
        sample_end = 0

        proxy_e1, proxy_e2 = model.aggregation(kg1, kg2)
        # print(proxy_e1.unsqueeze(0))
        _, suf_rank = torch.mm(proxy_e1.unsqueeze(0).cuda(), self.embed[self.split:].t()).sort(descending=True)
        retain = torch.where(suf_rank[0] == e1)[0][0]
        proxy_e1, proxy_e2 = model.aggregation(f_kg1, f_kg2)
        _, nec_rank = torch.mm(proxy_e1.unsqueeze(0).cuda(), self.embed[self.split:].t()).sort(descending=True)
        dec = torch.where(nec_rank[0] == e1)[0][0]
        
        
        if retain < 2 and dec > 0:
            sample_end = 1
        
        
        if soft_match == 1 and sample_end == 0:
            soft_num = 5
            
            
            sort_index = self.semantic_match_soft(p_embed1[f_kg1], p_embed2[f_kg2])
            add = sort_index[: soft_num]
            for cur in add:
                idx1 = int(cur / len(f_kg2))
                idx2 = int(cur % len(f_kg2))
                # print(idx1)
                # print(len(f_kg1))
                t1 = f_kg1[idx1]
                t2 = f_kg2[idx2]
                exp_pair.append([t1, t2])
                '''
                cur1 = [e_dict_r[t1[0]], r_dict_r[t1[1]], e_dict_r[t1[2]]]
                cur2 = [e_dict_r[t2[0]], r_dict_r[t2[1]], e_dict_r[t2[2]]]
                print(self.G_dataset.ent_dict[cur1[0]], self.G_dataset.r_dict[cur1[1]], self.G_dataset.ent_dict[cur1[2]])
                print(self.G_dataset.ent_dict[cur2[0]], self.G_dataset.r_dict[cur2[1]], self.G_dataset.ent_dict[cur2[2]])
                print('----------------------')
                '''
        i = 0
        k = 1
        mask = []
        Y = []
        comb_num = 2
        p = exp_pair
        
        while i < 1000 and k <= min(comb_num, int(len(p) / 2) +1):
            comb = [list(c)  for c in combinations(list(range(len(p))), k)]
            
            for coal in comb:
                time1 = time.time()
                tri1 = []
                tri2 = []
                no_tri1 = []
                no_tri2 = []
                cur_mask1 = [0] * (len(p))
                cur_mask2 = [1] * (len(p))
                # cur_mask1 = [0] * (len(new_p1) + len(new_p2))
                # cur_mask2 = [1] * (len(new_p1) + len(new_p2))
                cur_p = p.copy()
                for cur in coal:
                    tri1.append(p[cur][0])
                    tri2.append(p[cur][1])
                    cur_p.remove(p[cur])
                    cur_mask1[cur] = 1
                    cur_mask2[cur] = 0
                for pair in cur_p:
                    no_tri1.append(p[cur][0])
                    no_tri2.append(p[cur][1])
                # print(tri1, tri2)
                # print(no_tri1, no_tri2)
                try:
                    proxy_e1, proxy_e2 = model.aggregation(tri1, tri2)
                    v_suf = F.cosine_similarity(proxy_e1, proxy_e2, dim=0)
                
                    mask.append(cur_mask1)
                    Y.append(v_suf)
                    if len(no_tri1) and len(no_tri2):
                        proxy_e1, proxy_e2 = model.aggregation(no_tri1, no_tri2)
                        v_nec = F.cosine_similarity(proxy_e1, proxy_e2, dim=0)

                        mask.append(cur_mask2)
                    
                        Y.append(v_nec)
                    i += 1
                        # time2 = time.time()
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print('WARNING: out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        else:
                            raise exception
                    # print(time2 - time1, v_suf, v_nec)
            k += 1
        
        Z = torch.Tensor(mask)
        Y = torch.Tensor(Y)
        I = torch.eye(Z.shape[1])
        res = torch.mm(torch.inverse(torch.mm(Z.t(),Z) + I), torch.mm(Z.t(), Y.unsqueeze(1)))
        # print(Z)
        res = res.squeeze(1)
        score, indices = res.sort(descending=True)
        tri1 = set()
        tri2 = set()
        cur_tri1 = []
        cur_tri2 = []
        no_tri1 = new_p1.copy()
        no_tri2 = new_p2.copy()
        mask1 = [0] * (len(p1) + len(p2))
        mask2 = [1] * (len(p1) + len(p2))
        print('select pair len: ', len(p))
        for i in range(len(p)):
            if i > 11:
                break
            if score[i] < 0:
                continue

            t1 = new_p1[p[indices[i]][0]]
            t2 = new_p2[p[indices[i]][1]]
            if mask1[p[indices[i]][0]] == 0:
                mask1[p[indices[i]][0]] = 1
                cur_tri1.append(t1)
            if mask1[p[indices[i]][1]] == 0:
                mask1[p[indices[i]][1]] = 1
                cur_tri2.append(t2)
            if mask2[p[indices[i]][0]] == 1:
                mask2[p[indices[i]][0]] = 0
                no_tri1.remove(t1)
            if mask2[p[indices[i]][1]] == 1:
                mask2[p[indices[i]][1]] = 0
                no_tri2.remove(t2)
        
            cur1 = [e_dict_r[t1[0]], r_dict_r[t1[1]], e_dict_r[t1[2]]]
            tri1.add((e_dict_r[t1[0]], r_dict_r[t1[1]], e_dict_r[t1[2]]))
            tri2.add((e_dict_r[t2[0]], r_dict_r[t2[1]], e_dict_r[t2[2]]))
            
            cur2 = [e_dict_r[t2[0]], r_dict_r[t2[1]], e_dict_r[t2[2]]]
            print(self.G_dataset.ent_dict[cur1[0]], self.G_dataset.r_dict[cur1[1]], self.G_dataset.ent_dict[cur1[2]])
            print(self.G_dataset.ent_dict[cur2[0]], self.G_dataset.r_dict[cur2[1]], self.G_dataset.ent_dict[cur2[2]])
            # if self.judge(e1, e2, model, cur_tri1, cur_tri2, no_tri1, no_tri2, e_dict, r_dict) == True:
                # break
            print(score[i])
      
        print(len(tri1), len(tri2))
        return self.change_to_list(tri1), self.change_to_list(tri2)

    def realign(self):
        L = self.L[:15000]
        R = self.R[:15000]
        sim = torch.mm(L, R.t())
        # print(sim.shape)
        hit1 = 0
        right_pair = set()
        for i in range(15000):
            rank = (-sim[i, :]).argsort()
            rank_index = torch.where(rank == i)
            # print(rank_index)
            # print(rank)
            if rank[0] == i:
                hit1 += 1
                right_pair.add((i, i + 19388))

        print(hit1 / 15000)
        with open('../datasets/dbp_z_e/mtranse_no_train', 'w') as f:
            for cur in right_pair:
                f.write(str(cur[0]) + '\t' + str(cur[1]) + '\n')

    def sim_add(self, gid1, gid2, neigh1, neigh2):
        e1 = self.embed[int(gid1)]
        e2 = self.embed[int(gid2)]
        # neigh1 = []
        # neigh2 = []
        # print(self.L.shape)
        sim = torch.mm(e2.unsqueeze(0), self.L.t())
        rank = (-sim[0, :]).argsort()
        l = max(len(neigh1), len(neigh2))
        for i in range(self.L.shape[0]):
            # path_len = nx.dijkstra_path_length(self.G_dataset.G, source=int(gid1), target=int(rank[i]))
            # if path_len <= 2:
            if int(rank[i]) in self.G_dataset.all_2_hop1[gid1] and int(rank[i]) not in neigh1:
                neigh1.append(int(rank[i]))
                if len(neigh1) == 2 * l:
                    break
        sim = torch.mm(e1.unsqueeze(0), self.R.t())
        # l = 3 * len(neigh2)
        rank = (-sim[0, :]).argsort()
        for i in range(self.R.shape[0]):
            # path_len = nx.dijkstra_path_length(self.G_dataset.G, source=int(gid2), target=int(rank[i] + 19388))
            # if path_len <= 2:
            # print(rank[i])
            if int(rank[i] + 19388) in self.G_dataset.all_2_hop2[gid2] and int(rank[i] + 19388) not in neigh2 and int(rank[i] + 19388) != gid2:
                neigh2.append(int(rank[i] + 19388))
                if len(neigh2) == 2 * l:
                    break
        l_filter = min(len(neigh1), len(neigh2))
        neigh1 = set(neigh1[:l_filter])
        neigh2 = set(neigh2[:l_filter])
        
        
        return neigh1, neigh2


    def sim_add1(self, gid1, gid2, neigh1, neigh2):
        e1 = self.embed[int(gid1)]
        e2 = self.embed[int(gid2)]
        print(self.L.shape)
        sim = torch.mm(e2.unsqueeze(0), self.L.t())
        rank = (-sim[0, :]).argsort()
        l = len(neigh1)
        print(l)
        for i in range(self.L.shape[0]):
            # path_len = nx.dijkstra_path_length(self.G_dataset.G, source=int(gid1), target=int(rank[i]))
            # if path_len <= 2:
            if int(rank[i]) in self.G_dataset.all_2_hop1[gid1]:
                neigh1.add(int(rank[i]))
                if len(neigh1) == 2 * l:
                    break
        sim = torch.mm(e1.unsqueeze(0), self.R.t())
        # l = 3 * len(neigh2)
        rank = (-sim[0, :]).argsort()
        for i in range(self.R.shape[0]):
            # path_len = nx.dijkstra_path_length(self.G_dataset.G, source=int(gid2), target=int(rank[i] + 19388))
            # if path_len <= 2:
            # print(rank[i])
            if int(rank[i] + 19388) in self.G_dataset.all_2_hop2[gid2]:
                neigh2.add(int(rank[i] + 19388))
                if len(neigh2) == 2 * l:
                    break

        return neigh1, neigh2

    
    def init_1_hop(self, gid1, gid2):
        neigh1 = set()
        neigh2 = set()
        for cur in self.G_dataset.gid[gid1]:
            if cur[0] != int(gid1):
                neigh1.add(int(cur[0]))
            else:
                neigh1.add(int(cur[2]))
        for cur in self.G_dataset.gid[gid2]:
            if cur[0] != int(gid2):
                neigh2.add(int(cur[0]))
            else:
                neigh2.add(int(cur[2]))
        return neigh1, neigh2

    '''
    def search_2_hop_tri(self, source ,target):
        tri = set()
        for t in self.G_dataset.gid[source]:
            if t[0] == target or t[2] == target:
                tri.add((t[0], t[1], t[2]))
                continue
            if t[0] != source:
                for t2 in self.G_dataset.gid[t[0]]:
                    # print(t[0],target, t2)
                    if t2[0] == target or t2[2] == target:
                        tri.add(((t[0], t[1], t[2]),(t2[0], t2[1], t2[2])))
            else:
                for t2 in self.G_dataset.gid[t[2]]:
                    if t2[0] == target or t2[2] == target:
                        tri.add(((t[0], t[1], t[2]),(t2[0], t2[1], t2[2])))
        return tri
    '''
    def search_1_hop_tri(self, source ,target):
        tri = set()
        for t in self.G_dataset.gid[source]:
            if t[0] == target or t[2] == target:
                tri.add((t[0], t[1], t[2]))
                continue

        return tri
    
    def give_explain_set(self, gid, neigh):
        tri = set()
        for cur in neigh:
            tri |= self.search_2_hop_tri(gid, cur)
        return tri



    def show_explain(self, gid1, gid2, neigh1, neigh2):
        print(neigh1, neigh2)
        tri_set1 = self.give_explain_set(gid1, neigh1)
        tri_set2 = self.give_explain_set(gid2, neigh2)


        for cur in tri_set1:
            print('--------------------------------')
            if len(cur) == 3:
                print(self.G_dataset.ent_dict[cur[0]], self.G_dataset.r_dict[cur[1]], self.G_dataset.ent_dict[cur[2]])
            else:

                print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.r_dict[cur[0][1]], self.G_dataset.ent_dict[cur[0][2]])
                print(self.G_dataset.ent_dict[cur[1][0]], self.G_dataset.r_dict[cur[1][1]], self.G_dataset.ent_dict[cur[1][2]])
        for cur in tri_set2:
            print('--------------------------------')
            if len(cur) == 3:
                print(self.G_dataset.ent_dict[cur[0]], self.G_dataset.r_dict[cur[1]], self.G_dataset.ent_dict[cur[2]])
            else:
                print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.r_dict[cur[0][1]], self.G_dataset.ent_dict[cur[0][2]])
                print(self.G_dataset.ent_dict[cur[1][0]], self.G_dataset.r_dict[cur[1][1]], self.G_dataset.ent_dict[cur[1][2]])


    def stable_match(self, neigh_pre1, neigh_pre2):
        l = neigh_pre1.shape[0]
        is_kg1_finish = [False] * l
        is_kg2_finish = [False] * l
        res = [-1] * l 
        while False in is_kg1_finish:
            cur1 = is_kg1_finish.index(False)
            pre = neigh_pre1[cur1]
            for cur2 in pre:
                if is_kg2_finish[cur2] == False:
                    res[cur1] = int(cur2)
                    is_kg2_finish[cur2] = True
                    is_kg1_finish[cur1] = True
                    break
                else:
                    cur_pre = res.index(cur2)
                    # if neigh_pre2[cur2].index(cur_pre) >  neigh_pre2[cur2].index(cur1):
                    if torch.where(neigh_pre2[cur2] == cur_pre)[0][0] >  torch.where(neigh_pre2[cur2] == cur1)[0][0]:
                        is_kg1_finish[cur_pre] = False
                        res[cur_pre] = -1
                        is_kg1_finish[cur1] = True
                        res[cur1] = int(cur2)
                        break
        return res


    def max_weight_match(self, neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, sim, thred):
        G = nx.Graph()
        edges = []
        # print(sim.shape)
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                # print(sim[i][neigh_pre1[i][j]])
                if sim[i][neigh_pre1[i][j]] >= thred:
                    edges.append(( neigh_list2[int(neigh_pre1[i][j])],neigh_list1[i], sim[i][neigh_pre1[i][j]]))
                
                    # edges.append((int(neigh_pre1[i][j]), i, sim[i][neigh_pre1[i][j]]))
                else:
                    break
        G.add_weighted_edges_from(edges)
        return sorted(nx.max_weight_matching(G))


    def explain_pair_mask(self, gid1, gid2, pair):
        pair = self.change_to_list(pair)
        explainer = ExplainPair(pair, self.embed)
        explainer = explainer.cuda()
        optimizer = torch.optim.Adam( explainer.parameters(), lr=0.01, weight_decay=0)
        # print(explainer.triple_mask.requires_grad)
        explainer.train()

        for epoch in range(1000):
            optimizer.zero_grad()
            loss= explainer()
            # print(loss)
            loss.backward()
            optimizer.step()
        res_neigh = set()
        res = explainer.mask
        pair = torch.Tensor(pair)
        res = pair[torch.where(res > 0)]
        return res
        # print(neigh_list1, neigh_list2)
        '''
        for cur in res:
            if cur[0] < 19388:
                res_neigh.add(((cur[0],cur[1]), 1))
            else:
                res_neigh.add(((cur[1],cur[0]), 1))
        
        return res_neigh
        '''

    def explain_mask(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)

        explainer = ExplainModelGraph(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1))
        explainer = explainer.cuda()
        optimizer = torch.optim.Adam( explainer.parameters(), lr=0.01, weight_decay=0)
        # print(explainer.triple_mask.requires_grad)
        explainer.train()

        for epoch in range(1000):
            optimizer.zero_grad()
            loss= explainer()
            # print(loss)
            loss.backward()
            optimizer.step()

        res = torch.Tensor(list(range(len(p1) + len(p2))))[torch.where(explainer.mask > 0)]
        tri1 = []
        tri2 = []
        for cur in res:
            if cur < len(p1):
                tri1.append(p1[int(cur)])
            else:
                tri2.append(p2[int(cur) - len(p1)])
        return tri1, tri2
        

        # print(neigh_list1, neigh_list2)
        '''
        for cur in res:
            if cur[0] < 19388:
                res_neigh.add(((cur[0],cur[1]), 1))
            else:
                res_neigh.add(((cur[1],cur[0]), 1))
        
        return res_neigh
        '''


    def explain_rl(self, gid1, gid2):
        neigh1, neigh2 = self.init_1_hop(gid1, gid2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        if self.mapping is not None:
            neigh1_embed = self.mapping[neigh_list1]
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            neigh1_embed = self.embed[neigh_list1]
        neigh2_embed = self.embed[neigh_list2]
        act1 = torch.cat((neigh1_embed, torch.zeros(neigh1_embed.shape[0], 1).to(device)) , dim = 1).to(device)
        act2 = torch.cat((neigh2_embed, torch.ones(neigh2_embed.shape[0], 1).to(device)) , dim = 1).to(device)
        # act1 = torch.cat((neigh1_embed,  self.embed[gid2].repeat(neigh1_embed.shape[0], 1)), dim = 1).to(device)
        # act2 = torch.cat((neigh2_embed, self.embed[gid1].repeat(neigh2_embed.shape[0], 1)), dim = 1).to(device)
        # act = torch.cat((neigh1_embed, neigh2_embed), dim = 0)
        act = torch.cat((act1, act2), dim = 0)
        explainer = RL_one_hop(self.embed, gid1, gid2, act, neigh1_embed, neigh2_embed)
        explainer = explainer.cuda()
        optimizer = torch.optim.Adam(explainer.parameters(), lr=0.01, weight_decay=0)
        # print(explainer.triple_mask.requires_grad)
        explainer.train()
        # mask1 = torch.zeros(len(neigh_list1)).to(device)
        # mask2 = torch.zeros(len(neigh_list2)).to(device)
        for epoch in range(1000):
            mask1 = torch.zeros(len(neigh_list1)).to(device)
            mask2 = torch.zeros(len(neigh_list2)).to(device)
            explainer.reset()
            for i in range(int((min(len(neigh_list1), len(neigh_list2)) / 2) + 1)):
                optimizer.zero_grad()
                loss, mask1, mask2= explainer(mask1, mask2)
                print(loss, mask1, mask2)
                loss.backward()
                optimizer.step()
        
        # exit(0)
        res_neigh = set()
        res1 = mask1
        res2 = mask2
        neigh_list1 = torch.Tensor(neigh_list1)
        neigh_list2 = torch.Tensor(neigh_list2)
        # print(res1, res2)
        res1 = neigh_list1[torch.where(res1 > 0)]
        res2 = neigh_list2[torch.where(res2 > 0)]
        return res1, res2

    def change_to_list(self, exp):
        exp_list = []
        for cur in exp:
            exp_list.append(list(cur))
        return exp_list


    def compute_coal_value(self, gid1, gid2, c):
        c = torch.Tensor(c)
        if self.mapping is not None:
            neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        else:
            neigh1 = self.embed[c.t()[0].long()].mean(dim = 0)
        neigh1 = F.normalize(neigh1, dim = 0)
        neigh2 = self.embed[c.t()[1].long()].mean(dim = 0)
        neigh2 = F.normalize(neigh2, dim = 0)
        return F.cosine_similarity(neigh1, neigh2, dim=0)

    def analyze_rule(self, rule):
        need = set()
        delete = set()
        for condition in rule:
            tmp = condition.split(' ')
            if tmp[1] == '>' and int(float(tmp[2])) == 0:
                need.add(int(float(tmp[0])))
            elif tmp[1] == '<' and int(float(tmp[2])) == 1:
                delete.add(int(float(tmp[0])))
            elif tmp[1] == '>=' and int(float(tmp[2])) == 1:
                need.add(int(float(tmp[0])))
            elif tmp[1] == '==' and int(float(tmp[2])) == 1:
                need.add(int(float(tmp[0])))
            elif tmp[1] == '==' and int(float(tmp[2])) == 0:
                delete.add(int(float(tmp[0])))
            elif tmp[1] == '<=' and int(float(tmp[2])) == 0:
                delete.add(int(float(tmp[0])))
        return need, delete

    def analyze_rule_lore(self, rule, class_name):
        need = set()
        delete = set()
        if class_name == 1:
            for condition in rule:
                tmp = rule[condition]
                if tmp[0] == '>' and tmp[1] == '-':
                    continue
                elif tmp[0] == '<' and tmp[1] == '-':
                    delete.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and tmp[2] == '-':
                    delete.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    continue
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    delete.add(int(float(condition)))
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and tmp[2] == '-':
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    need.add(int(float(condition)))
                elif tmp[0] == '>' and int(float(tmp[1])) == 0:
                    need.add(int(float(condition)))
                elif tmp[0] == '<' and int(float(tmp[1])) == 1:
                    delete.add(int(float(condition)))
                elif tmp[0] == '=' and int(float(tmp[2])) == 1:
                    need.add(int(float(condition)))
                elif tmp[0] == '==' and int(float(tmp[1])) == 0:
                    delete.add(int(float(condition)))
        else:
            for condition in rule:
                tmp = rule[condition]
                if tmp[0] == '>' and tmp[1] == '-':
                    continue
                elif tmp[0] == '<' and tmp[1] == '-':
                    need.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and tmp[2] == '-':
                    need.add(int(float(condition)))
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    continue
                elif tmp[0] == '<' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    need.add(int(float(condition)))
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 0:
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and tmp[2] == '-':
                    continue
                elif tmp[0] == '>' and tmp[1] == '=' and int(float(tmp[2])) == 1:
                    delete.add(int(float(condition)))
                elif tmp[0] == '>' and int(float(tmp[1])) == 0:
                    delete.add(int(float(condition)))
                elif tmp[0] == '<' and int(float(tmp[1])) == 1:
                    need.add(int(float(condition)))
                elif tmp[0] == '=' and int(float(tmp[2])) == 1:
                    delete.add(int(float(condition)))
                elif tmp[0] == '==' and int(float(tmp[1])) == 0:
                    need.add(int(float(condition)))

        return need, delete

    def explain_anchor(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        explain = Anchor(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed)
        rule = explain.compute(100, 0.6)
        need, delete = self.analyze_rule(rule)
        need_tri = []
        delete_tri = []
        i = 0
        for cur in need:
            if cur < len(p1):
                need_tri.append(p1[cur])  
            else:
                need_tri.append(p2[cur - len(p1)])
        for cur in delete:
            if cur < len(p1):
                delete_tri.append(p1[cur])  
            else:
                delete_tri.append(p2[cur - len(p1)])
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule

    def explain_lore(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        if (len(p1) + len(p2)) < 50:
            explain = LORE(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed, self.lang)
            rule, class_name = explain.compute(30, 0.5)
        else:
            rule = None
        if rule == None:
            need_tri = []
            delete_tri = []
        else:
            need, delete = self.analyze_rule_lore(rule, class_name)
            need_tri = []
            delete_tri = []
            i = 0
            for cur in need:
                if cur < len(p1):
                    need_tri.append(p1[cur])  
                else:
                    need_tri.append(p2[cur - len(p1)])
            for cur in delete:
                if cur < len(p1):
                    delete_tri.append(p1[cur])  
                else:
                    delete_tri.append(p2[cur - len(p1)])
        need_rule = 'need:'
        for cur in need_tri:
            need_rule += '\t' + str(cur)
        delete_rule = 'delete:'
        for cur in delete_tri:
            delete_rule += '\t' + str(cur)
        rule = need_rule + ',' + delete_rule
        return rule

    def explain_lime(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        lime = LIME(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed)
        res = lime.compute(100)
        # res = res.squeeze(1)
        res = torch.Tensor(res)
        score, indices = res.sort(descending=True)
        tri1 = []
        tri2 = []
        tri = []
        i = 0
        for cur in indices:
            if cur < len(p1):
                tri.append(p1[cur])
                # self.read_triple_name(p1[cur])
                # print(score[i])   
            else:
                tri.append(p2[cur - len(p1)])
                # self.read_triple_name(p2[cur - len(p1)])
                # print(score[i]) 
            i += 1
        # return tri[:num]
        tri.append((0,0,0))
        return tri
    
    def explain_llm(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        llm = LLM(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1), e1, e2, self.embed)
        sim_num = int(600 / (len(p1) + len(p2)))
        Z, Y = llm.compute(sim_num)
        return Z, Y, p1, p2

    def explain_shapely(self, e1, e2):
        p1, p_embed1 = self.pattern_process(e1, 1)
        p2, p_embed2 = self.pattern_process(e2, 1)
        model, e_dict, r_dict, e_dict_r, r_dict_r, graph = self.get_proxy_model_ori(e1, e2)
        p_embed1 = p_embed1 / (torch.linalg.norm(p_embed1, dim=-1, keepdim=True) + 1e-5)
        p_embed2 = p_embed2 / (torch.linalg.norm(p_embed2, dim=-1, keepdim=True) + 1e-5)
        new_p1 = self.change_pattern_id(p1, e_dict, r_dict)
        new_p2 = self.change_pattern_id(p2, e_dict, r_dict)
        # print(p1, len(p1))
        Shapley = Shapley_Value(model, len(p1) + len(p2), list(range(len(p1) + len(p2))), len(p1))
        shapley_value = Shapley.MTC(100)
        res = torch.Tensor(shapley_value).argsort(descending=True)
        tri1 = []
        tri2 = []
        tri = []
        for cur in res:
            if cur < len(p1):
                tri.append(p1[cur])
                # self.read_triple_name(p1[cur])
                # print(shapley_value[cur])
            else:
                tri.append(p2[cur - len(p1)])
                # self.read_triple_name(p2[cur - len(p1)])
                # print(shapley_value[cur])
        # return tri[:num]
        tri.append((0,0,0))
        return tri
        # print(shapley_value)
        # exit(0)


    def Shapely_value_debug(self, exp, gid1, gid2):
        exp = self.change_to_list(exp)
        num_exp = len(exp)

        all_coal = [list(coal) for i in range(num_exp) for coal in combinations(exp, i + 1)]
        # print(num_exp)
        shapely_value = []
        for e in exp:
            # print(e)
            e_coal = []
            no_e_coal = []
            
            for c in copy.deepcopy(all_coal):
                if e in c:
                    value = self.compute_coal_value(gid1, gid2, c)
                    e_coal.append((copy.deepcopy(c), value))
                    c.remove(e)
                    if len(c) == 0:
                        no_e_coal.append((c, 0))
                    else:
                        value = self.compute_coal_value(gid1, gid2, c)
                        no_e_coal.append((c, value))
            # print('e联盟: ', e_coal)
            # print('noe联盟: ', no_e_coal)
            shapelyvalue = 0
            for i in range(len(e_coal)):
                s = len(e_coal[i][0])
                e_payoff = e_coal[i][1] - no_e_coal[i][1]
                e_weight = math.factorial(s-1)*math.factorial(num_exp-s)/math.factorial(num_exp)
                shapelyvalue += e_payoff * e_weight
            shapely_value.append((e,shapelyvalue))
        # print('夏普利值：',shapely_value)
        # print(len(shapely_value))
        shapely_value.sort(key=lambda x :x[1], reverse=True)
        for cur in shapely_value:
            print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.ent_dict[cur[0][1]])


    def Shapely_value(self, exp, gid1, gid2, suf=True):
        exp = self.change_to_list(exp)
        num_exp = len(exp)

        if num_exp > 10:
            exp = exp[:10]
            num_exp = 10
        all_coal = [list(coal) for i in range(num_exp) for coal in combinations(exp, i + 1)]
        
        print(num_exp)
        shapely_value = []
        
            
        for e in exp:
            # print(e)
            e_coal = []
            no_e_coal = []
            
            for c in copy.deepcopy(all_coal):
                if e in c:
                    if suf:
                        value = self.compute_coal_value(gid1, gid2, c)
                    else:
                        tmp_exp = copy.deepcopy(exp)
                        for cur in c:
                            tmp_exp.remove(cur)
                        if len(tmp_exp) == 0:
                            # no_e_coal.append((0, 0))
                            value = 0
                        else:
                            value = -self.compute_coal_value(gid1, gid2, tmp_exp)
                    l = len(c)
                    e_coal.append((l, value))
                    c.remove(e)
                    if len(c) == 0:
                        no_e_coal.append((0, 0))
                    else:
                        if suf:
                            value = self.compute_coal_value(gid1, gid2, c)
                        else:
                            tmp_exp = copy.deepcopy(exp)
                            for cur in c:
                                tmp_exp.remove(cur)
                            if len(tmp_exp) == 0:
                                # no_e_coal.append((0, 0))
                                value = 0
                            else:
                                value = -self.compute_coal_value(gid1, gid2, tmp_exp)
                        no_e_coal.append((l - 1, value))
            # print('e联盟: ', e_coal)
            # print('noe联盟: ', no_e_coal)
            shapelyvalue = 0
            for i in range(len(e_coal)):
                s = e_coal[i][0]
                e_payoff = e_coal[i][1] - no_e_coal[i][1]
                e_weight = math.factorial(s-1)*math.factorial(num_exp-s)/math.factorial(num_exp)
                shapelyvalue += e_payoff * e_weight
            shapely_value.append((e,shapelyvalue))
        # print('夏普利值：',shapely_value)
        # print(len(shapely_value))
        shapely_value.sort(key=lambda x :x[1], reverse=True)
        new_exp = []
        for cur in shapely_value:
            new_exp.append((cur[0][0], cur[0][1]))
            # if len(new_exp) > 4:
                # break
            print(self.G_dataset.ent_dict[cur[0][0]], self.G_dataset.ent_dict[cur[0][1]])
        return new_exp        


    def fine_tune_explain(self, neigh1, neigh2, gid1, gid2):
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        print(neigh_list1)
        neigh_list1.sort()
        neigh_list2.sort()
        if self.mapping is not None:
            neigh1_embed = self.mapping[torch.Tensor(neigh_list1).long()]
        else:
            neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
        neigh2_embed = self.embed[torch.Tensor(neigh_list2).long()]
        # print(neigh1_embed.shape)
        neigh_sim1 = torch.mm(neigh1_embed, neigh2_embed.t())
        neigh_sim2 = torch.mm(neigh2_embed, neigh1_embed.t())
       
        neigh_pre1 = (-neigh_sim1).argsort()
        neigh_pre2 = (-neigh_sim2).argsort()
        res = self.max_weight_match(neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, neigh_sim1, 0)
        res_neigh = set()
        for cur in res:
            if cur[0] < 19388:
                res_neigh.add((cur[0],cur[1]))
            else:
                res_neigh.add((cur[1],cur[0]))
        return res_neigh

    def explain_match(self, gid1, gid2):
        neigh1, neigh2 = self.init_1_hop(gid1, gid2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        # print(neigh2)
        # neigh1, neigh2 = self.sim_add(gid1, gid2, neigh_list1, neigh_list2)
        # neigh_list1 = list(neigh1)
        # neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        # print(neigh_list1)
        # print(neigh_list2)
        d1 = {}
        d2 = {}
        for i in range(len(neigh_list1)):
            d1[neigh_list1[i]] = i
        for i in range(len(neigh_list2)):
            d2[neigh_list2[i]] = i
        
        if self.mapping is not None:
            neigh1_embed = self.mapping[neigh_list1]
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            neigh1_embed = self.embed[neigh_list1]
        neigh2_embed = self.embed[neigh_list2]
        # print(neigh1_embed.shape)
        # neigh_sim1 = torch.mm(neigh1_embed, neigh2_embed.t())
        # neigh_sim2 = torch.mm(neigh2_embed, neigh1_embed.t())
        neigh_sim1 = self.cosine_matrix(neigh1_embed, neigh2_embed)
        neigh_sim2 = self.cosine_matrix(neigh2_embed, neigh1_embed)
        neigh_pre1 = (-neigh_sim1).argsort()
        neigh_pre2 = (-neigh_sim2).argsort()
        res = self.max_weight_match(neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, neigh_sim1, 0.75)
        sim_list = []
        # print(neigh_sim1)
        # print(neigh_sim11)
        for i in range(len(res)):
            cur = res[i]
            if cur[0] < 19388:
                sim_list.append((i,neigh_sim1[d1[cur[0]]][d2[cur[1]]]))
                # print(neigh_sim1[d1[cur[0]]][d2[cur[1]]])
            else:
                sim_list.append((i,neigh_sim1[d1[cur[1]]][d2[cur[0]]]))
                # print(neigh_sim1[d1[cur[1]]][d2[cur[0]]])
        res_neigh = list()
        # print(res)
        sim_list.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(sim_list)):
            cur = res[sim_list[i][0]]
            if cur[0] < 19388:
                res_neigh.append((cur[0],cur[1]))
                print(neigh_sim1[d1[cur[0]]][d2[cur[1]]])
            else:
                res_neigh.append((cur[1],cur[0]))
                print(neigh_sim1[d1[cur[1]]][d2[cur[0]]])
        # exit(0)
        return res_neigh
        # print(neigh1, neigh2)
        '''
        res_neigh = set()
        print(res)
        for cur in res:
            if cur[0] < 19388:
                res_neigh.add((cur[0],cur[1]))
            else:
                res_neigh.add((cur[1],cur[0]))
        return res_neigh
        '''

    def explain_base(self, gid1, gid2):
        neigh1, neigh2 = self.init_1_hop(gid1, gid2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        # print(neigh2)
        # neigh1, neigh2 = self.sim_add(gid1, gid2, neigh_list1, neigh_list2)
        # neigh_list1 = list(neigh1)
        # neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        # print(neigh_list1)
        # print(neigh_list2)
        # neigh1_embed = self.embed[neigh_list1]
        if self.mapping is not None:
            neigh1_embed = self.mapping[neigh_list1]
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            neigh1_embed = self.embed[neigh_list1]
        neigh2_embed = self.embed[neigh_list2]
        # print(neigh1_embed.shape)
        neigh_sim1 = torch.mm(neigh1_embed, neigh2_embed.t())
        neigh_sim2 = torch.mm(neigh2_embed, neigh1_embed.t())
       
        neigh_pre1 = (-neigh_sim1).argsort()
        neigh_pre2 = (-neigh_sim2).argsort()
        # res = self.max_weight_match(neigh_pre1, neigh_pre2, neigh_list1, neigh_list2, neigh_sim1, 0.75)
        res = []
        for i in range(neigh_pre1.shape[0]):
            res.append(((int(neigh_list1[i]),int(neigh_list2[int(neigh_pre1[i][0])])), neigh_sim1[i][neigh_pre1[i][0]]))
        # print(neigh1, neigh2)
        res.sort(key=lambda x :x[1], reverse=True)
        res_neigh = []
        print(res)
        for cur in res:
            if cur[0][0] < 19388:
                res_neigh.append((cur[0][0],cur[0][1]))
            else:
                res_neigh.append((cur[0][1],cur[0][0]))
        return res_neigh


    def explain_stable(self, gid1, gid2):
        neigh1, neigh2 = self.init_1_hop(gid1, gid2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        # print(neigh2)
        neigh1, neigh2 = self.sim_add(gid1, gid2, neigh_list1, neigh_list2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        # print(neigh_list1)
        # neigh1_embed = self.embed[neigh_list1]
        if self.mapping is not None:
            neigh1_embed = self.mapping[neigh_list1]
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            neigh1_embed = self.embed[neigh_list1]
        neigh2_embed = self.embed[neigh_list2]
        # print(neigh1_embed.shape)
        neigh_sim1 = torch.mm(neigh1_embed, neigh2_embed.t())
        neigh_sim2 = torch.mm(neigh2_embed, neigh1_embed.t())
        # print(neigh_sim1)
        # print(neigh_sim2)
        '''
        for cur in neigh_list1:
            print(self.G_dataset.ent_dict[int(cur)])
        for cur in neigh_list2:
            print(self.G_dataset.ent_dict[int(cur)])
        '''
        neigh_pre1 = (-neigh_sim1).argsort()
        neigh_pre2 = (-neigh_sim2).argsort()
        res = self.stable_match(neigh_pre1, neigh_pre2)
        
        # print(res)
        neigh2 = torch.Tensor(neigh_list2)
        neigh1 = torch.Tensor(neigh_list1)
        neigh2 = neigh2[res]
        neigh_sim = {}
        for i in range(neigh1.shape[0]):
            neigh_sim[int(neigh1[i]), int(neigh2[i])] =  neigh_sim1[i][res[i]]
            # print(self.G_dataset.ent_dict[int(neigh1[i])], self.G_dataset.ent_dict[int(neigh2[i])], neigh_sim1[i][res[i]])
        # exit(0)
        res_neigh = sorted(neigh_sim.items(), key = lambda d: d[1], reverse = True)[:int(neigh1.shape[0] / 2)]
        return res_neigh

    def cosine_matrix(self, A, B):
        A_sim = torch.mm(A, B.t())
        a = torch.norm(A, p=2, dim=-1)
        b = torch.norm(B, p=2, dim=-1)
        cos_sim = A_sim / a.unsqueeze(-1)
        cos_sim /= b.unsqueeze(-2)
        return cos_sim

    def sim_dist_test(self, gid1, gid2):
        neigh1 = set()
        neigh2 = set()
        for cur in self.G_dataset.gid1[gid1]:
            if cur[0] != gid1:
                neigh1.add(cur[0])
            else:
                neigh1.add(cur[2])
        for cur in self.G_dataset.gid2[gid2]:
            if cur[0] != gid2:
                neigh2.add(cur[0])
            else:
                neigh2.add(cur[2])
        embed = self.embed
        x = []
        y = []
        tri1 = []
        # comb1 = list(combinations(list(neigh1), 2))
        for cur1 in neigh1:
            for cur2 in neigh2:
                e11 = embed[int(cur1)] / (torch.linalg.norm(embed[int(cur1)], dim=-1, keepdim=True) + 1e-5)
                e21 = embed[int(cur2)] / (torch.linalg.norm(embed[int(cur2)], dim=-1, keepdim=True) + 1e-5)
                neigh_embed1 = embed[list(neigh1 - {cur1})].mean(dim = 0)
                neigh_embed2 = embed[list(neigh2 - {cur2})].mean(dim = 0)
                # neigh_embed2 = embed[list(neigh2)].mean(dim = 0)
                e12 = neigh_embed1 / (torch.linalg.norm(neigh_embed1, dim=-1, keepdim=True) + 1e-5)
                e22 = neigh_embed2 / (torch.linalg.norm(neigh_embed2, dim=-1, keepdim=True) + 1e-5)
                # e1 = (e11 + e12) / 2
                # d1 = float(F.pairwise_distance(e1, e2, p=2))
                # d2 = float(F.pairwise_distance(e11, e2, p=2))
                # d3 = float(F.pairwise_distance(e12, e2, p=2))
                d1 = float(F.cosine_similarity(e11, e21, dim=0))
                d2 = float(F.cosine_similarity(e12, e22, dim=0))
                # d3 = float(F.cosine_similarity(e12, e2, dim=0))
                x.append(d1)
                y.append(d2)
                print(self.G_dataset.ent_dict[int(cur1)], self.G_dataset.ent_dict[int(cur2)])
                print(d1, d2)
                print(d1 + d2)
            plt.scatter(x,y)
            plt.savefig('zh-en/embed_nec_suf_2_' + str(gid1) + '.jpg')
            plt.show()
            plt.clf()
        # exit(0)

            
    def explain_greedy(self, gid1, gid2):
        neigh1, neigh2 = self.init_1_hop(gid1, gid2)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        neigh1 = torch.Tensor(neigh_list1)
        neigh2 = torch.Tensor(neigh_list2)
        print(neigh_list1)
        print(neigh_list2)
        # neigh1_embed = self.embed[neigh_list1]
        if self.mapping is not None:
            neigh1_embed = self.mapping[neigh_list1]
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            neigh1_embed = self.embed[neigh_list1]
        neigh2_embed = self.embed[neigh_list2]
        neigh_sim1 = torch.mm(neigh1_embed, neigh2_embed.t())
        print(neigh_sim1)
        neigh_sim_sort = list((-torch.mm(neigh1_embed, neigh2_embed.mean(dim=0).unsqueeze(0).t()).squeeze(1)).sort().indices)
        cur_neigh = []
        for i in range(len(neigh_sim_sort)):
            if i >= len(neigh_list2):
                break
            cur_neigh.append(neigh_list1[neigh_sim_sort[i]])
            print(neigh_sim_sort[i])
            print(cur_neigh)
            nec1 = neigh_sim_sort[i+1:]
            cur_neigh1 = self.embed[cur_neigh].mean(dim=0).unsqueeze(0)
            neigh_sim_sort2 = list((-torch.mm(neigh2_embed, cur_neigh1.t()).squeeze(1)).sort().indices)
            suf2 = neigh_sim_sort2[:len(cur_neigh)]
            nec2 = neigh_sim_sort2[len(cur_neigh):]
            # print(suf2)
            # print(neigh2)
            # print(neigh2[nec2])
            cur_neigh2 = self.embed[neigh2[torch.Tensor(suf2).long()].long()].mean(dim=0).unsqueeze(0)
            se1 = cur_neigh1 
            se2 = cur_neigh2
            # se1 = cur_neigh1 / (torch.linalg.norm(cur_neigh1, dim=-1, keepdim=True) + 1e-5)
            # se2 = cur_neigh2 / (torch.linalg.norm(cur_neigh2, dim=-1, keepdim=True) + 1e-5)
            
            cur_neigh_nec1 = self.embed[neigh1[torch.Tensor(nec1).long()].long()].mean(dim=0).unsqueeze(0)
            cur_neigh_nec2 = self.embed[neigh2[torch.Tensor(nec2).long()].long()].mean(dim=0).unsqueeze(0)
            # ne1 = cur_neigh_nec1 / (torch.linalg.norm(cur_neigh_nec1, dim=-1, keepdim=True) + 1e-5)
            # ne2 = cur_neigh_nec2 / (torch.linalg.norm(cur_neigh_nec2, dim=-1, keepdim=True) + 1e-5)
            ne1 = cur_neigh_nec1
            ne2 = cur_neigh_nec2 
            s1 = F.cosine_similarity(se1.squeeze(0), se2.squeeze(0), dim=0)
            s2 = F.cosine_similarity(ne1.squeeze(0), ne2.squeeze(0), dim=0)
            if s1 > 0.8 and s2 < 0.9:
                return cur_neigh, list(neigh2[torch.Tensor(suf2).long()])
            # print(se1, se2, ne1, ne2)
            print(s1, s2)
        # exit(0)
        return cur_neigh, list(neigh2[torch.Tensor(suf2).long()])

    def f(self, pair, mask):
        if self.mapping is not None:
            # neigh1_embed = torch.matmul(self.embed[neigh_list1], self.mapping)
            me1 = (mask.unsqueeze(1) * self.mapping[pair.T[0,:].long()]).mean(dim = 0)
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            # neigh1_embed = self.embed[neigh_list1]
            me1 = (mask.unsqueeze(1) * self.embed[pair.T[0,:].long()]).mean(dim = 0)
        me2 = (mask.unsqueeze(1) * self.embed[pair.T[1,:].long()]).mean(dim = 0)
        if self.mapping is not None:
            ume1 = ((1 - mask).unsqueeze(1) *  self.mapping[pair.T[0,:].long()]).mean(dim = 0)
        else:
            ume1 = ((1 - mask).unsqueeze(1) * self.embed[pair.T[0,:].long()]).mean(dim = 0)
        ume2 = ((1 - mask).unsqueeze(1) * self.embed[pair.T[1,:].long()]).mean(dim = 0)
        me1 = F.normalize(me1, dim = 0)
        me2 = F.normalize(me2, dim = 0)
        ume1 = F.normalize(ume1, dim = 0)
        ume2 = F.normalize(ume2, dim = 0)
        d1 = F.pairwise_distance(me1, me2, p=2).to(device)  # 让变动跟原本的尽可能近
        if torch.count_nonzero(mask) == mask.shape[0]:
            d1 = 2
        d2 = F.pairwise_distance(ume1, ume2, p=2).to(device) # 让变动后跟原本的尽可能远
        # print(mask, d1-d2,pair_num / pair.shape[0])
        # return alpha * (d1 - d2) + (1 - alpha) * (1 - (pair_num / pair.shape[0]))
        return  (1 + d2 - d1) 

    def compute_margin(self, pair, mask, j):
        f1 = self.f(pair, mask)
        new_mask = mask.clone()
        new_mask[j] = (not new_mask[j])
        f2 = self.f(pair, new_mask)
        print(f2 - f1)
        return f2 - f1

    def explain_greedy_search(self,gid1, gid2, pair, k):
        pair = self.change_to_list(pair)
        pair = torch.Tensor(pair)
        T = torch.zeros(pair.shape[0])
        mask = torch.zeros(pair.shape[0]).cuda()
        best_mask = None
        best_val = -1e10
        for i in range(2 * k):
            T += 1
            delta = torch.zeros(pair.shape[0])
            judge = torch.zeros(pair.shape[0])
            for j in range(pair.shape[0]):
                # mask[j] = (not mask[j])
                delta[j] = self.compute_margin(pair, mask,j) 
                judge[j] = 100 * delta[j] + T[j]
            x = judge.argmax()
            if delta[x] > 0 or torch.count_nonzero(mask) <= k:
                mask[x] = (not mask[x])
                T[x] = 0
                value = self.compute_margin(pair, mask, x)
                if value > best_val:
                    best_mask = mask
                    best_val = value
            else:
                judge = torch.zeros(pair.shape[0])
                for j in range(pair.shape[0]):
                    if mask[j] == 1:
                        judge[j] = 100 * self.compute_margin(pair, mask, j) + T[j]
                    else:
                        judge[j] = -1e10
                x = judge.argmax()
                mask[x] = (not mask[x])
        res = pair[torch.where(best_mask > 0)]
        return res
                    

    def test_case(self, file, e1, e2):
        nec_tri = set()
        nec_ent = set()
        with open(file) as f:
            lines = f.readlines()
            for cur in lines:
                cur = cur.strip().split('\t')
                nec_tri.add((int(self.G_dataset.id_ent[cur[0]]), int(self.G_dataset.id_r[cur[1]]), int(self.G_dataset.id_ent[cur[2]])))
                nec_ent.add(int(self.G_dataset.id_ent[cur[0]]))
                nec_ent.add(int(self.G_dataset.id_ent[cur[2]]))
        gid1 = int(self.G_dataset.id_ent[e1])
        gid2 = int(self.G_dataset.id_ent[e2])
        print(gid1)
        neigh1, neigh2 = self.init_1_hop(gid1, gid2)
        print(neigh1)
        neigh_list1 = list(neigh1)
        neigh_list2 = list(neigh2)
        # print(neigh2)
        # neigh1, neigh2 = self.sim_add(gid1, gid2, neigh_list1, neigh_list2)
        # neigh_list1 = list(neigh1)
        # neigh_list2 = list(neigh2)
        neigh_list1.sort()
        neigh_list2.sort()
        # print(neigh_list1)
        # print(neigh_list2)
        # neigh1_embed = self.embed[neigh_list1]
        if self.mapping is not None:
            neigh1_embed = self.mapping[neigh_list1]
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            neigh1_embed = self.embed[neigh_list1]
        neigh2_embed = self.embed[neigh_list2]
        s1 = self.cosine_matrix(neigh2_embed, neigh1_embed)
        s2 = self.cosine_matrix(neigh2_embed, neigh2_embed)
        print(s1, s2)
        mask1 = torch.zeros(len(neigh1)).to(device) 
        mask2 = torch.zeros(len(neigh2)).to(device) 
        for i in range(len(neigh_list1)):
            if neigh_list1[i] in nec_ent:
                mask1[i] = 1
        for i in range(len(neigh_list2)):
            if neigh_list2[i] in nec_ent:
                mask2[i] = 1
        # print(mask1)
        # print(neigh1_embed)
        me1 = (mask1.unsqueeze(1) * neigh1_embed).mean(dim = 0)
        me2 = (mask2.unsqueeze(1) * neigh2_embed).mean(dim = 0)
        print(F.cosine_similarity(me1, me2, dim=0))

        # print(F.pairwise_distance(me1, me2, p=2))

            
            

    def explain_sa(self, gid1, gid2, pair):
        
        pair = self.change_to_list(pair)
        pair = torch.Tensor(pair)
        
        sim0 = torch.cosine_similarity(self.embed[pair.T[0,:].long()], self.embed[pair.T[1,:].long()], dim=-1)
        print(sim0)
        mask = torch.zeros(pair.shape[0]).to(device)
        # mask[0] = 1
        # cur_val = self.compute_cur_val(pair, mask)
        # pro = F.softmax(sim, dim = 0)
        # iter_num = 1000
        # j = 0
        best_mask = None
        gold_mask = None
        cur_best = -1e10
        gold_best = -1e10
        l = list(range(pair.shape[0]))
        for j in range(pair.shape[0]):
            cur_rel = []
            for c in combinations(l, j + 1):
                index = torch.Tensor(list(c)).long()
                # print(index)
                cur_rel.append((index, sim0[index].mean(dim=0)))
            cur_rel.sort(key=lambda x : x[1], reverse=True)
            for cur in cur_rel:
                mask[cur[0]] = 1
                cur_val = self.compute_cur_val(pair, mask)
                if gold_mask == None:
                    gold_mask = mask.clone()
                    gold_best = cur_val
                else:
                    if gold_best < cur_val:
                        gold_mask = mask.clone()
                        gold_best = cur_val
                    elif cur_val / gold_best < random.uniform(0,1):
                        break
                # print(cur_val, mask)
                if gold_best > 1.2:
                    return pair[torch.where(gold_mask > 0)]

        
        if best_mask == None:
            best_mask = gold_mask
        print(best_mask)
        res = pair[torch.where(best_mask > 0)]
        return res
    

    def explain_combinations(self, gid1, gid2, pair):
        pair = self.change_to_list(pair)
        pair = torch.Tensor(pair)
        # sim0 = torch.cosine_similarity(self.embed[pair.T[0,:].long()], self.embed[pair.T[1,:].long()], dim=-1)

        mask = torch.zeros(pair.shape[0]).to(device)
        # mask[0] = 1
        # cur_val = self.compute_cur_val(pair, mask)
        # pro = F.softmax(sim, dim = 0)
        # iter_num = 1000
        # j = 0
        best_mask = None
        gold_mask = None
        cur_best = -1e10
        gold_best = -1e10
        l = list(range(pair.shape[0]))
        for j in range(pair.shape[0]):
            for c in combinations(l, j + 1):
                mask[torch.Tensor(list(c)).long()] = 1
                cur_val = self.compute_cur_val(pair, mask)
                if gold_mask == None:
                    gold_mask = mask.clone()
                    gold_best = cur_val
                else:
                    if gold_best < cur_val:
                        gold_mask = mask.clone()
                        gold_best = cur_val
        
        if best_mask == None:
            best_mask = gold_mask
        print(best_mask)
        res = pair[torch.where(best_mask > 0)]
        return res

    def compute_cur_val(self, pair, mask):
        if self.mapping is not None:
            # neigh1_embed = torch.matmul(self.embed[neigh_list1], self.mapping)
            me1 = (mask.unsqueeze(1) * self.mapping[pair.T[0,:].long()]).mean(dim = 0)
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            # neigh1_embed = self.embed[neigh_list1]
            me1 = (mask.unsqueeze(1) * self.embed[pair.T[0,:].long()]).mean(dim = 0)
        me2 = (mask.unsqueeze(1) * self.embed[pair.T[1,:].long()]).mean(dim = 0)
        if self.mapping is not None:
            ume1 = ((1 - mask).unsqueeze(1) *  self.mapping[pair.T[0,:].long()]).mean(dim = 0)
        else:
            ume1 = ((1 - mask).unsqueeze(1) * self.embed[pair.T[0,:].long()]).mean(dim = 0)
        ume2 = ((1 - mask).unsqueeze(1) * self.embed[pair.T[1,:].long()]).mean(dim = 0)
        me1 = F.normalize(me1, dim = 0)
        me2 = F.normalize(me2, dim = 0)
        ume1 = F.normalize(ume1, dim = 0)
        ume2 = F.normalize(ume2, dim = 0)
        d1 = F.pairwise_distance(me1, me2, p=2).to(device)  # 让变动跟原本的尽可能近
        d2 = F.pairwise_distance(ume1, ume2, p=2).to(device) # 让变动后跟原本的尽可能远
        pair_num = torch.count_nonzero(mask)
        alpha = 0.999
        # print(mask, d1-d2,pair_num / pair.shape[0])
        # return alpha * (d1 - d2) + (1 - alpha) * (1 - (pair_num / pair.shape[0]))
        return 1 + alpha * (d2-d1) + (1 - alpha) * (1 - (pair_num / pair.shape[0]))


    def explain_random_walk(self, gid1, gid2, pair):
        pair = self.change_to_list(pair)
        pair = torch.Tensor(pair)
        sim0 = torch.cosine_similarity(self.embed[pair.T[0,:].long()], self.embed[pair.T[1,:].long()], dim=-1)

        mask = torch.zeros(pair.shape[0]).to(device)
        mask[0] = 1
        cur_val = self.compute_cur_val(pair, mask)
        # pro = F.softmax(sim, dim = 0)
        iter_num = 1000
        j = 0
        best_mask = None
        gold_mask = None
        cur_best = -1e10
        gold_best = -1e10
        while(j < iter_num):
            j += 1
            sim = sim0.clone()
            whether_add = random.uniform(0,1)
            pair_num = torch.count_nonzero(mask)
            # print(whether_add)
            if whether_add > pair_num / pair.shape[0]:
                sim[torch.where(mask > 0)] = -1e10
                pro = F.softmax(sim, dim = 0)
                cur_p = 0
                random_pro = random.uniform(0,1)
                for i in range(len(pro)):
                    cur_p += pro[i]
                    if cur_p >= random_pro:
                        mask[i] = 1
                        break
            else:
                sim[torch.where(mask == 0)] = 1e10
                pro = F.softmax(-sim, dim = 0)
                # print(pro)
                cur_p = 0
                random_pro = random.uniform(0,1)
                for i in range(len(pro)):
                    cur_p += pro[i]
                    if cur_p >= random_pro:
                        mask[i] = 0
                        break
            pair_num = torch.count_nonzero(mask)
            if pair_num == 0:
                cur_val = -1e10
            else:
                cur_val = self.compute_cur_val(pair, mask)
            # print(gold_best, cur_val,gold_mask,mask)
            if gold_mask == None:
                gold_mask = mask.clone()
            else:
                if gold_best < cur_val:
                    gold_mask = mask.clone()
                    gold_best = cur_val

            # print(pair_num , pair.shape[0])
            if pair_num / pair.shape[0] <= 0.5 and pair_num != 0:
                if best_mask == None:
                    best_mask = mask.clone()
                else:
                    if cur_best < cur_val:
                        best_mask = mask.clone()
                        cur_best = cur_val
                # print(mask)
                # print(cur_val)
        
        if best_mask == None:
            best_mask = gold_mask
        print(best_mask)
        res = pair[torch.where(best_mask > 0)]
        return res
        # exit(0)
            

           
    



class ExplainPair(torch.nn.Module):
    def __init__(self, pair, embed, mapping=None):
        super(ExplainPair, self).__init__()
        self.pair = torch.Tensor(pair)
        self.embed = embed
        self.mask= self.construct_mask()
        self.mapping = mapping

    def forward(self):
        mask = self.get_masked_triple()
        print(mask)
        # print(mask.shape, self.pair.shape)
        # print(self.embed[self.pair.T[0:].long()].shape)
        if self.mapping is not None:
            # neigh1_embed = torch.matmul(self.embed[neigh_list1], self.mapping)
            me1 = (mask.unsqueeze(1) * self.mapping[pair.T[0,:].long()]).mean(dim = 0)
        else:
            # neigh1_embed = self.embed[torch.Tensor(neigh_list1).long()]
            # neigh1_embed = self.embed[neigh_list1]
            me1 = (mask.unsqueeze(1) * self.embed[pair.T[0,:].long()]).mean(dim = 0)
        me2 = (mask.unsqueeze(1) * self.embed[pair.T[1,:].long()]).mean(dim = 0)
        if self.mapping is not None:
            ume1 = ((1 - mask).unsqueeze(1) *  self.mapping[pair.T[0,:].long()]).mean(dim = 0)
        else:
            ume1 = ((1 - mask).unsqueeze(1) * self.embed[pair.T[0,:].long()]).mean(dim = 0)
        ume2 = ((1 - mask).unsqueeze(1) * self.embed[pair.T[1,:].long()]).mean(dim = 0)
        me1 = F.normalize(me1, dim = 0)
        me2 = F.normalize(me2, dim = 0)
        ume1 = F.normalize(ume1, dim = 0)
        ume2 = F.normalize(ume2, dim = 0)
        l1 = torch.linalg.norm(mask, ord=1) / mask.shape[0]
        # l2 = torch.linalg.norm(mask2, ord=1) / mask2.shape[0]
        d1 = F.pairwise_distance(me1, me2, p=2).to(device)  # 让变动跟原本的尽可能近
        d2 = F.pairwise_distance(ume1, ume2, p=2).to(device) # 让变动后跟原本的尽可能远
        relu = torch.nn.ReLU()
        alpha = 1
        # print(d2 , d1, l1)
        # print(l1 + l2, l1, l2)
        loss = (1 - alpha) * relu(0.5 - d2) + alpha * d1 
        # print(d2 , d1, l1)
        # print(mask1, mask2)
        return loss



    def construct_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.pair.shape[0]), requires_grad=True)

        
        std1 = torch.nn.init.calculate_gain("relu") * math.sqrt(
            1 / (self.pair.shape[0])
        )

        with torch.no_grad():
            mask.normal_(1.0, std1)

        
        return mask

    def get_masked_triple(self):
        
        mask = torch.sigmoid(self.mask)
        return mask
    


class ExplainModelGraph(torch.nn.Module):
    def __init__(self, model, num_players, players, split):
        super(ExplainModelGraph, self).__init__()
        self.model = model
        self.num_players = num_players
        self.players = players
        self.split = split
        self.mask = self.construct_mask()


    def early_stop(self, d5, d6, len1, len2):
        if self.cur_stat == None:
            self.cur_stat = (d5, d6, len1, len2)
        else:
            if len1 < len(self.tri1) /2 and len2 < len(self.tri2) /2:
                return True
            self.cur_stat = (d6, len1, len2)
        return False


    def rank(self, e, candidate):
        #print(e)
        sim = torch.mm(e, candidate.t()).squeeze(0)
        # print(sim)
        rank_index = sim.topk(k =1, dim=0).indices[0]
        return rank_index, sim


    def forward(self):
        mask = self.get_masked_triple()
        # print('--------cur mask-------------')
        # print(mask1,mask2)
        sim1 = self.model.mask_sim(mask, self.split)
        sim2 = self.model.mask_sim(1 - mask, self.split)
        relu = torch.nn.ReLU()
        alpha = 0.5
        # loss = sim2
        # print(l1 + l2, l1, l2)
        # loss = sim2
        loss = (1 - alpha) * relu(1 - sim1) + alpha * sim2 
        # print(d2 , d1)
        # print(mask1, mask2)
        return loss



    def construct_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_players), requires_grad=True)
        # mask2 = torch.nn.Parameter(torch.FloatTensor(len(self.neigh2)), requires_grad=True)

        
        std1 = torch.nn.init.calculate_gain("relu") * math.sqrt(
            1 / (self.num_players)
        )
        with torch.no_grad():
            mask.normal_(1.0, std1)
        
        return mask

    def get_masked_triple(self):
        
        mask = torch.sigmoid(self.mask)
        return mask
    
    def get_explain(self):
        # exp = self.triple_mask - self.rel_fact
        
        exp = torch.sigmoid(self.triple_mask)  > self.exp_thred
        
        return exp
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(f'arguments for Explanation Generation or Entity Alignment Repair')

    parser.add_argument('lang', type=str, help='which dataset', default='zh')
    parser.add_argument('method', type=str, help='Explanation Generation or Entity Alignment Repair', default='repair')
    parser.add_argument('--version', type=str, help='the hop num of candidate neighbor', default=1)
    parser.add_argument('--num', type=str, help='the len of explanation', default=15)
    
    args = parser.parse_args()
    lang = args.lang
    method = args.method
    if args.version:
        version = args.version
    if args.num:
        num = int(args.num)

    pair = '/pair'


    device = 'cuda'
    if lang == 'zh':
        G_dataset = DBpDataset('../datasets/dbp_z_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_z_e/' + pair)
    elif lang == 'ja':
        G_dataset = DBpDataset('../datasets/dbp_j_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_j_e/' + pair)
    elif lang == 'fr':
        G_dataset = DBpDataset('../datasets/dbp_f_e/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_f_e/' + pair)
    elif lang == 'y':
        G_dataset = DBpDataset('../datasets/D_Y/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/D_Y/' + pair)
    elif lang == 'w':
        G_dataset = DBpDataset('../datasets/D_W/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/D_W/' + pair)
    elif lang == 'w2':
        G_dataset = DBpDataset('../datasets/D_W2/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/D_W2/' + pair)
    elif lang == 'zh2':
        G_dataset = DBpDataset('../datasets/z_e_2/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/z_e_2/' + pair)
    elif lang == 'zh_2':
        G_dataset = DBpDataset('../datasets/dbp_z_e_wrong/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_z_e_wrong/' + pair)
    elif lang == 'ja2':
        G_dataset = DBpDataset('../datasets/dbp_j_e_wrong/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_j_e_wrong/' + pair)
    elif lang == 'fr2':
        G_dataset = DBpDataset('../datasets/dbp_f_e_wrong/', device=device, pair=pair, lang=lang)
        test_indices = read_list('../datasets/dbp_f_e_wrong/' + pair)
    Lvec = None
    Rvec = None
    model_name = 'mean_pooling'
    saved_model = None
    args = None
    in_d = None
    out_d = None
    # lang = 'zh'

    m_adj=None
    e1=None
    e2=None
    device = 'cuda'
    if lang == 'zh':
        split = len(read_list('../datasets/dbp_z_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_z_e/rel_dict1'))
    elif lang == 'ja':
        split = len(read_list('../datasets/dbp_j_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_j_e/rel_dict1'))
    elif lang == 'fr':
        split = len(read_list('../datasets/dbp_f_e/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_f_e/rel_dict1'))
    elif lang == 'y':
        split = len(read_list('../datasets/D_Y/ent_dict1'))
        splitr = len(read_list('../datasets/D_Y/rel_dict1'))
    elif lang == 'w':
        split = len(read_list('../datasets/D_W/ent_dict1'))
        splitr = len(read_list('../datasets/D_W/rel_dict1'))
    elif lang == 'w2':
        split = len(read_list('../datasets/D_W2/ent_dict1'))
        splitr = len(read_list('../datasets/D_W2/rel_dict1'))
    elif lang == 'zh2':
        split = len(read_list('../datasets/z_e_2/ent_dict1'))
        splitr = len(read_list('../datasets/z_e_2/rel_dict1'))
    elif lang == 'zh_2':
        split = len(read_list('../datasets/dbp_z_e_wrong/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_z_e_wrong/rel_dict1'))
    elif lang == 'ja2':
        split = len(read_list('../datasets/dbp_j_e_wrong/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_j_e_wrong/rel_dict1'))
    elif lang == 'fr2':
        split = len(read_list('../datasets/dbp_f_e_wrong/ent_dict1'))
        splitr = len(read_list('../datasets/dbp_f_e_wrong/rel_dict1'))

    model = None
    evaluator = None
    model_name = 'load'
    model_name = 'mean_pooling'
    explain = EAExplainer(model_name, G_dataset, test_indices, Lvec, Rvec, model, evaluator, split, splitr, lang)
    explain.explain_EA(method,0.4, num,  version)
    
