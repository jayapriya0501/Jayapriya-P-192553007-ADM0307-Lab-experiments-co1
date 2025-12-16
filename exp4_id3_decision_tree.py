import pandas as pd
import numpy as np
data = pd.read_csv('data/decision_tree_data.csv')
def entropy(y): p = y.value_counts(normalize=True); return -sum(p * np.log2(p + 1e-9))
def info_gain(data, attr, target):
    total_ent = entropy(data[target])
    vals, counts = np.unique(data[attr], return_counts=True)
    weighted_ent = sum((counts[i]/sum(counts)) * entropy(data[data[attr]==vals[i]][target]) for i in range(len(vals)))
    return total_ent - weighted_ent
def id3(data, attrs, target, depth=0):
    if len(data[target].unique()) == 1: return data[target].iloc[0]
    if not attrs: return data[target].mode()[0]
    gains = {a: info_gain(data, a, target) for a in attrs}
    best = max(gains, key=gains.get)
    tree = {best: {v: id3(data[data[best]==v], [a for a in attrs if a!=best], target, depth+1) for v in data[best].unique()}}
    return tree
print("ID3 Decision Tree:", id3(data, list(data.columns[:-1]), 'PlayTennis'))
