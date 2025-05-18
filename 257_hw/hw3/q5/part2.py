
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Part 1
p = 0.7
Ns = [1, 50, 1000]
num_trials = 500
np.random.seed(42)  # Reproducibility

# Parameters for Part 2
epsilon = 0.1
max_N = 100
num_trials_per_N = 10

Ns_part2 = np.arange(1, max_N + 1)
proportions = []
hoeffding = []

for n in Ns_part2:
    # Compute empirical probability
    samples = np.random.binomial(1, p, size=(num_trials_per_N, n))
    sample_means = samples.mean(axis=1)
    diffs = np.abs(sample_means - p)
    count = np.sum(diffs >= epsilon)
    proportions.append(count / num_trials_per_N)
    
    # Compute Hoeffding bound
    hoeffding_bound = 2 * np.exp(-2 * n * (epsilon ** 2))
    hoeffding.append(min(hoeffding_bound, 1.0))  # Cap at 1

# Plotting
plt.figure()
plt.plot(Ns_part2, proportions, label='Empirical Probability')
plt.plot(Ns_part2, hoeffding, label='Hoeffding Bound')
plt.title('Empirical Probability vs Hoeffding Bound (ε=0.1)')
plt.xlabel('N')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.savefig('/root/code/ucsd_hw/257_hw/hw3/figure/q5_part2_plot.png')
plt.show()