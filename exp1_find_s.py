import pandas as pd
data = pd.read_csv('data/concept_data.csv')
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
h = ['0'] * X.shape[1]
for i, label in enumerate(y):
    if label == 'Yes':
        for j in range(len(h)):
            h[j] = X[i][j] if h[j] == '0' else ('?' if h[j] != X[i][j] else h[j])
print("Final Hypothesis:", h)
