import numpy as np
np.random.seed(42)
true_mean, true_std = 5.0, 2.0
data = np.random.normal(true_mean, true_std, 1000)
mle_mean = np.mean(data)
mle_std = np.sqrt(np.mean((data - mle_mean) ** 2))
print(f"True parameters: mean={true_mean}, std={true_std}")
print(f"MLE estimates: mean={mle_mean:.4f}, std={mle_std:.4f}")
bernoulli_data = np.random.binomial(1, 0.7, 500)
mle_p = np.mean(bernoulli_data)
print(f"Bernoulli MLE (true p=0.7): p={mle_p:.4f}")
