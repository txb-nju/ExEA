from transformers import BertTokenizer, BertModel
import torch

# bert = BertModel.from_pretrained('/data/xbtian/bert-base-uncased', output_hidden_states=True)
# tokenizer = BertTokenizer.from_pretrained('/data/xbtian/bert-base-uncased')

def bert_encode(bert, tokenizer, name):
    encoded_input = tokenizer(name, return_tensors='pt')
    output = bert(**encoded_input)
    x = output['last_hidden_state'][0].mean(dim = 0)
    return x
    
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

def read_link1(file):
    d1 = {}
    d2 = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split(' ')
            name = cur[1]
            for i in range(len(cur) - 2):
                name += ' ' + cur[i + 2]
            # print(name)
            d1[cur[0]] = name
            d2[name] = cur[0]
    return d1, d2

def read_link2(file):
    d1 = {}
    d2 = {}
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            cur = line.strip().split(' ')
            name = cur[2]
            print(cur)
            if len(cur) > 3:
                for i in range(len(cur) - 3):
                    name += ' ' + cur[i + 3]
            print(name)
            d1[cur[0]] = name
            d2[name] = cur[0]
    return d1, d2

def cosine_matrix(A, B):
    A_sim = torch.mm(A, B.t())
    a = torch.norm(A, p=2, dim=-1)
    b = torch.norm(B, p=2, dim=-1)
    cos_sim = A_sim / a.unsqueeze(-1)
    cos_sim /= b.unsqueeze(-2)
    return cos_sim

d1, d2 = read_link('rel_dict1')
d3, d4 = read_link('rel_dict2')
d5, d6 = read_link1('rel_name_2')
d7, d8 =  read_link2('rel_name_link2')
with open('rel_links', 'w') as f:
    for cur in d7:
        print(d7[cur])
        if cur ==  'rdf-schema#seeAlso':
            f.write(d2['http://www.w3.org/2000/01/rdf-schema#seeAlso'] + '\t' +  d4[d6[d7[cur]]] + '\n')
        elif cur == 'owl#differentFrom':
            f.write(d2['http://www.w3.org/2002/07/owl#differentFrom'] + '\t' +  d4[d6[d7[cur]]] + '\n')
        else:
            f.write(d2['http://dbpedia.org/ontology/' + cur] + '\t' +  d4[d6[d7[cur]]] + '\n')
'''
embed1 = []
embed2 = []
name1 = []
name2 = []
for cur in d1:
    name = d1[cur].split('/')[-1]
    embed = bert_encode(bert, tokenizer, name).tolist()
    embed1.append(embed)
    name1.append(name)

for cur in d5:
    name = d5[cur]
    embed = bert_encode(bert, tokenizer, name).tolist()
    embed2.append(embed)
    name2.append(name)

embed1 = torch.Tensor(embed1)
embed2 = torch.Tensor(embed2)
torch.save(embed1, 'rel1.pt')
torch.save(embed2, 'rel2.pt')
embed1 =  torch.load('rel1.pt')
embed2 =  torch.load('rel2.pt')
sim_l =cosine_matrix(embed1, embed2)
sim_r =cosine_matrix(embed2, embed1)
rankl = (-sim_l).argsort()
rankr = (-sim_r).argsort()
for i in range(rankl.shape[0]):
    print(name1[i], name2[rankl[i][0]])
'''


