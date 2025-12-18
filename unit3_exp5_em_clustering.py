import numpy as np
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [3, 3], np.random.randn(50, 2) + [-3, -3]])
K, n = 2, len(X)
means = X[np.random.choice(n, K, replace=False)]
covs, weights = [np.eye(2)] * K, np.ones(K) / K
for _ in range(50):
    pdf = lambda x, m, c: np.exp(-0.5 * np.sum((x-m) @ np.linalg.inv(c) * (x-m), axis=1)) / np.sqrt(np.linalg.det(c))
    resp = np.array([weights[k] * pdf(X, means[k], covs[k]) for k in range(K)]).T
    resp /= resp.sum(axis=1, keepdims=True)
    for k in range(K):
        Nk = resp[:, k].sum()
        means[k] = (resp[:, k:k+1].T @ X) / Nk
        covs[k] = ((resp[:, k:k+1] * (X - means[k])).T @ (X - means[k])) / Nk
        weights[k] = Nk / n
print("Cluster centers:", means)
print("Cluster assignments:", resp.argmax(axis=1))
