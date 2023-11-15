with open('rel_dict2') as f:
    lines = f.readlines()
    for line in lines:
        cur = line.strip().split('\t')
        print(cur[1])