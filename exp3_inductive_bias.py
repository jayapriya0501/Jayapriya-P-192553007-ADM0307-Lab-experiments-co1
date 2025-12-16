import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, 50)
models = {'Linear (high bias)': LinearRegression(), 'Polynomial (medium)': LinearRegression(), 'DecisionTree (low bias)': DecisionTreeRegressor(max_depth=10)}
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
for name, model in models.items():
    X_fit = X_poly if 'Polynomial' in name else X
    model.fit(X_fit, y)
    print(f"{name}: Train R² = {model.score(X_fit, y):.4f}")
