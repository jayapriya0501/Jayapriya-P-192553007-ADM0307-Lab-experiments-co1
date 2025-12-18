import numpy as np
np.random.seed(42)
P_disease, P_pos_given_disease, P_pos_given_no_disease = 0.01, 0.99, 0.05
P_pos = P_pos_given_disease * P_disease + P_pos_given_no_disease * (1 - P_disease)
P_disease_given_pos = (P_pos_given_disease * P_disease) / P_pos
print(f"Prior P(Disease): {P_disease}")
print(f"P(Positive|Disease): {P_pos_given_disease}, P(Positive|No Disease): {P_pos_given_no_disease}")
print(f"P(Positive): {P_pos:.4f}")
print(f"Posterior P(Disease|Positive): {P_disease_given_pos:.4f}")
