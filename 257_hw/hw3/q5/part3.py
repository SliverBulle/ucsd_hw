from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
# Parameters for Part 3
mu_cont = 0.5  # Uniform [0,1] mean
Ns_cont = np.arange(1, 101)
epsilons_cont = np.linspace(0.01, 0.5, 20)
num_trials_cont = 100

# Create grid
N_grid, epsilon_grid = np.meshgrid(Ns_cont, epsilons_cont)
prob_grid = np.zeros_like(N_grid)
hoeffding_grid = 2 * np.exp(-2 * N_grid * epsilon_grid**2)

# Compute empirical probabilities
for i, epsilon in enumerate(epsilons_cont):
    for j, n in enumerate(Ns_cont):
        samples = np.random.uniform(0, 1, size=(num_trials_cont, n))
        sample_means = samples.mean(axis=1)
        diffs = np.abs(sample_means - mu_cont)
        prob_grid[i, j] = np.mean(diffs >= epsilon)

# 3D Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(N_grid, epsilon_grid, prob_grid, cmap='viridis', alpha=0.6, label='Empirical')
ax.plot_surface(N_grid, epsilon_grid, hoeffding_grid, cmap='plasma', alpha=0.6, label='Hoeffding')
ax.set_xlabel('N')
ax.set_ylabel('ε')
ax.set_zlabel('Probability')
ax.set_title('3D Plot of Empirical Probability and Hoeffding Bound')
plt.savefig('/root/code/ucsd_hw/257_hw/hw3/figure/q5_part3_plot.png')
plt.show()