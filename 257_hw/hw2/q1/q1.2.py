import numpy as np
import matplotlib.pyplot as plt

# define the objective function
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + np.sin(10*x[0]) * np.sin(10*x[1])

# SA
def simulated_annealing(f3,dim, iterations, seed):
    np.random.seed(seed)
    x = np.ones(dim)
    T0 = 1
    points = []

    for k in range(iterations):
        T = T0 * (0.95 ** k)
        new_x = x + np.random.normal(0, 0.1, dim)
        delta_f = f3(new_x) - f3(x)
        if delta_f < 0 or np.random.rand() < np.exp(-delta_f / T):
            x = new_x
        points.append(x.copy())

    return points

# CE
def cross_entropy_method(f3,dim, iterations, seed):
    np.random.seed(seed)
    mean = np.ones(dim)
    cov = np.eye(dim)
    points = []

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mean, cov, size=50)
        sample_values = np.array([f3(sample) for sample in samples])
        elite_samples = samples[np.argsort(sample_values)[:10]]  # 20% of 50 is 10
        mean = np.mean(elite_samples, axis=0)
        new_cov = np.cov(elite_samples, rowvar=False)
        if np.all(np.linalg.eigvals(new_cov) > 0):
            cov = new_cov
        points.append(mean.copy())

    return points

# search gradient method
def search_gradient_method(f3,dim, iterations, seed, step_size=0.02):
    np.random.seed(seed)
    mean = np.ones(dim)
    cov = 0.1 * np.eye(dim)
    points = []

    for _ in range(iterations):
        samples = np.random.multivariate_normal(mean, cov, size=50)
        sample_values = np.array([f3(sample) for sample in samples])
        grad = np.mean(2 * samples, axis=0)
        if np.linalg.norm(grad) > 10:
            grad = grad / np.linalg.norm(grad) * 10
        mean = mean - step_size * grad
        new_cov = np.cov(samples, rowvar=False)
        if np.all(np.linalg.eigvals(new_cov) > 0):
            cov = new_cov
        points.append(mean.copy())

    return points

# standard gradient descent method
def gradient_descent(f3,dim, iterations, seed, step_size=0.02):
    np.random.seed(seed)
    x = np.ones(dim)
    points = []

    for _ in range(iterations):
        if f3 == f1:
            grad = 2 * x
        elif f3 == f2:
            grad = 2 * x + 10 * np.cos(10 * x) * np.sin(10 * x)
        x = x - step_size * grad
        points.append(x.copy())

    return points
def compute_function_on_grid(func, X1, X2):
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = func([X1[i, j], X2[i, j]])
    return Z
# set parameters
dim = 2
iterations = 100
seeds = [42, 43, 44]
levels = np.linspace(0.1, 2, 20)

# plot the figure
functions = {
    "f1": f1,
    "f2": f2
}

methods = {
    "Simulated Annealing": simulated_annealing,
    "Cross-Entropy Method": cross_entropy_method,
    "Search Gradient Method": search_gradient_method,
    "Gradient Descent": gradient_descent
}

for func_name, func in functions.items():
    for seed in seeds:
        plt.figure(figsize=(8, 8))
        x1 = np.linspace(-2, 2, 400)
        x2 = np.linspace(-2, 2, 400)
        X1, X2 = np.meshgrid(x1, x2)
        Z = compute_function_on_grid(func, X1, X2)
        
        # plot the contour
        plt.contour(X1, X2, Z, levels=levels, colors='black', linewidths=0.5)

        # plot the points sequence of each algorithm
        for name, method in methods.items():
            points = method(func,dim, iterations, seed)
            points = np.array(points)
            plt.plot(points[:, 0], points[:, 1], label=f'{name} (Seed={seed})')

        plt.title(f'{func_name} Level Sets and Algorithm Paths (Seed={seed})')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.legend()
        plt.savefig(f"/root/code/257_hw/hw2/figure/q1.2_{func_name}_{seed}.png")