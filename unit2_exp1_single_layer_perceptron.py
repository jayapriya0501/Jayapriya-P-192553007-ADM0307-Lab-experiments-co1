import numpy as np
np.random.seed(42)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND gate
weights, bias, lr = np.random.randn(2), 0, 0.1
for epoch in range(100):
    for i in range(len(X)):
        pred = 1 if np.dot(X[i], weights) + bias > 0 else 0
        error = y[i] - pred
        weights += lr * error * X[i]
        bias += lr * error
print("Weights:", weights, "Bias:", bias)
print("Predictions:", [1 if np.dot(x, weights) + bias > 0 else 0 for x in X])
