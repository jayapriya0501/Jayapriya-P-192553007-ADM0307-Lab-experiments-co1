import numpy as np
np.random.seed(42)
X = np.vstack([np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [-2, -2]])
y = np.array([0]*50 + [1]*50)
classes = np.unique(y)
params = {c: {'mean': X[y==c].mean(axis=0), 'var': X[y==c].var(axis=0), 'prior': (y==c).mean()} for c in classes}
def predict(x):
    probs = {c: np.log(params[c]['prior']) + np.sum(-0.5*np.log(2*np.pi*params[c]['var']) - (x-params[c]['mean'])**2/(2*params[c]['var'])) for c in classes}
    return max(probs, key=probs.get)
preds = [predict(x) for x in X]
print(f"Accuracy: {np.mean(preds == y):.4f}")
