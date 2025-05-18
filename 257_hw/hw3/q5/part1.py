import numpy as np
import matplotlib.pyplot as plt

# Parameters for Part 1
p = 0.7
Ns = [1, 50, 1000]
num_trials = 500
np.random.seed(42)  # Reproducibility

for n in Ns:
    # Generate sample means for 500 trials
    samples = np.random.binomial(1, p, size=(num_trials, n))
    sample_means = samples.mean(axis=1)
    
    # Plot histogram
    plt.figure()
    plt.hist(sample_means, bins=np.arange(0, 1.1, 0.1), edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of $\\bar X_N$ for N={n}')
    plt.xlabel('$\\bar X_N$')
    plt.ylabel('Frequency')
    plt.xlim(0, 1)
    plt.grid(True)
    plt.savefig(f'/root/code/ucsd_hw/257_hw/hw3/figure/q5_part1_plot_{n}.png')
plt.show()