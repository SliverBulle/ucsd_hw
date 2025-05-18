import numpy as np
import matplotlib.pyplot as plt
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + np.sin(10*x[0]) * np.sin(10*x[1])
# define the objective function
def f3(x):
    return np.sum(x**2)

# SA
def simulated_annealing(dim, iterations, seed):
    np.random.seed(seed)
    x = np.ones(dim)
    T0 = 1
    values = []

    for k in range(iterations):
        T = T0 * (0.95 ** k)
        new_x = x + np.random.normal(0, 0.1, dim)
        delta_f = f3(new_x) - f3(x)
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / T):
            x = new_x
        values.append(f3(x))

    return values

# CE
def cross_entropy_method(dim, iterations, seed):
    np.random.seed(seed)
    mean = np.ones(dim)
    cov = np.eye(dim)
    values = []

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mean, cov, size=50)
        sample_values = np.array([f3(sample) for sample in samples])
        elite_samples = samples[np.argsort(sample_values)[:10]]  # 20% of 50 is 10
        mean = np.mean(elite_samples, axis=0)
        new_cov = np.cov(elite_samples, rowvar=False)
        if np.all(np.linalg.eigvals(new_cov) > 0):
            cov = new_cov  # only update the covariance matrix when it is positive definite
        values.append(np.mean(sample_values))

    return values

# search gradient method
def search_gradient_method(dim, iterations, seed, step_size=0.02):
    np.random.seed(seed)
    mean = np.ones(dim)
    cov = 0.1 * np.eye(dim)
    values = []

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mean, cov, size=50)
        sample_values = np.array([f3(sample) for sample in samples])
        grad = np.mean(2 * samples, axis=0)
        if np.linalg.norm(grad) > 10:
            grad = grad / np.linalg.norm(grad) * 10
        mean = mean - step_size * grad
        new_cov = np.cov(samples, rowvar=False)
        if np.all(np.linalg.eigvals(new_cov) > 0):
            cov = new_cov  # only update the covariance matrix when it is positive definite
        values.append(np.mean(sample_values))

    return values

# standard gradient descent method
def gradient_descent(dim, iterations, seed, step_size=0.02):
    np.random.seed(seed)
    x = np.ones(dim)
    values = []

    for _ in range(iterations):
        grad = 2 * x
        x = x - step_size * grad
        values.append(f3(x))

    return values

# set parameters
dim = 50
iterations = 100
seeds = [42, 43, 44]

# plot the figure
methods = {
    "Simulated Annealing": simulated_annealing,
    "Cross-Entropy Method": cross_entropy_method,
    "Search Gradient Method": search_gradient_method,
    "Gradient Descent": gradient_descent
}

for seed in seeds:
    plt.figure(figsize=(12, 8))
    for name, method in methods.items():
        values = method(dim, iterations, seed)
        plt.plot(values, label=f'{name} (Seed={seed})')
    plt.title(f'Function Value over Iterations (Seed={seed})')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.savefig(f'/root/code/257_hw/hw2/q1.3_{seed}.png')
