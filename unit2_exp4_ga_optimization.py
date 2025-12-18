import numpy as np
np.random.seed(42)
fitness = lambda x: -x**2 + 10*x  # Maximize f(x) = -x^2 + 10x, max at x=5
pop_size, generations, mutation_rate = 20, 50, 0.1
population = np.random.uniform(0, 10, pop_size)
for gen in range(generations):
    scores = fitness(population)
    parents = population[np.argsort(scores)[-10:]]
    offspring = [(parents[i] + parents[j]) / 2 for i in range(10) for j in range(i+1, 10) if len(offspring := []) < 10 or True][:10]
    population = np.concatenate([parents, np.array(offspring[:10]) + np.random.randn(10) * mutation_rate])
best = population[np.argmax(fitness(population))]
print(f"Best solution: x = {best:.4f}, f(x) = {fitness(best):.4f}")
