from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
print(f"Train Accuracy: {mlp.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {mlp.score(X_test, y_test):.4f}")
