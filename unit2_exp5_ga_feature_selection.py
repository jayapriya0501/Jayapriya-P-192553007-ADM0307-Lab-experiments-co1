import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
np.random.seed(42)
X, y = np.random.randn(100, 10), (np.random.randn(100) > 0).astype(int)
pop_size, n_features, generations = 20, 10, 20
population = np.random.randint(0, 2, (pop_size, n_features))
def fitness(mask): return cross_val_score(LogisticRegression(), X[:, mask.astype(bool)], y, cv=3).mean() if mask.sum() > 0 else 0
for _ in range(generations):
    scores = np.array([fitness(p) for p in population])
    parents = population[np.argsort(scores)[-10:]]
    children = np.array([parents[i] ^ ((np.random.rand(n_features) < 0.1).astype(int)) for i in range(10)])
    population = np.vstack([parents, children])
best = population[np.argmax([fitness(p) for p in population])]
print(f"Selected features: {np.where(best)[0]}, Score: {fitness(best):.4f}")
