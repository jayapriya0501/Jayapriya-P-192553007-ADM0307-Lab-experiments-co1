import numpy as np
np.random.seed(42)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # XOR gate
sigmoid = lambda x: 1 / (1 + np.exp(-x))
w1, w2, b1, b2, lr = np.random.randn(2, 4), np.random.randn(4, 1), np.zeros((1, 4)), np.zeros((1, 1)), 1
for _ in range(10000):
    h = sigmoid(X @ w1 + b1)
    o = sigmoid(h @ w2 + b2)
    d2 = (o - y) * o * (1 - o)
    d1 = (d2 @ w2.T) * h * (1 - h)
    w2 -= lr * h.T @ d2; b2 -= lr * d2.sum(axis=0)
    w1 -= lr * X.T @ d1; b1 -= lr * d1.sum(axis=0)
print("XOR Predictions:", (sigmoid(sigmoid(X @ w1 + b1) @ w2 + b2) > 0.5).astype(int).flatten())
