from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
nb = GaussianNB()
nb.fit(X_train, y_train)
print(f"Train Accuracy: {nb.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {nb.score(X_test, y_test):.4f}")
print(f"Class priors: {nb.class_prior_}")
