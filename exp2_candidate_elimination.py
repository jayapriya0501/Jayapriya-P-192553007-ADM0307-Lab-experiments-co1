import pandas as pd
data = pd.read_csv('data/concept_data.csv')
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
S, G = ['0'] * X.shape[1], [['?'] * X.shape[1]]
for i, label in enumerate(y):
    if label == 'Yes':
        for j in range(len(S)):
            S[j] = X[i][j] if S[j] == '0' else ('?' if S[j] != X[i][j] else S[j])
        G = [g for g in G if all(g[j] == '?' or g[j] == S[j] for j in range(len(S)))]
    else:
        G = [g[:j] + [S[j]] + g[j+1:] for g in G for j in range(len(S)) if S[j] != '0' and S[j] != X[i][j] and g[j] == '?']
print("Specific Boundary:", S)
print("General Boundary:", G)
