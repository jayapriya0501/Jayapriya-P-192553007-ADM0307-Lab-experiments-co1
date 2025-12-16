import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
np.random.seed(42)
X = np.random.randn(100, 10)
y = (X[:, 0] + X[:, 2] + X[:, 5] > 0).astype(int)
np.savetxt('data/heuristic_data.csv', np.column_stack([X, y]), delimiter=',', header=','.join([f'f{i}' for i in range(10)] + ['target']))
def heuristic_search(X, y, n_features=10):
    selected, remaining = [], list(range(n_features))
    for _ in range(3):
        scores = {f: cross_val_score(LogisticRegression(), X[:, selected + [f]], y, cv=3).mean() for f in remaining}
        best = max(scores, key=scores.get)
        selected.append(best); remaining.remove(best)
        print(f"Selected feature {best}, Score: {scores[best]:.4f}")
    return selected
print("Best features:", heuristic_search(X, y))
