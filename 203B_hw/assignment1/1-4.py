import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm
from scipy.sparse.linalg import gmres

def run_gmres(A, b, tol=1e-8):
    # initial guess
    x0 = np.zeros_like(b)
    
    # run GMRES
    x, exitCode = gmres(A, b, x0=x0, tol=tol, callback_type='pr_norm', callback=callback)
    
    return x

# store the residual of each iteration
residuals = []

def callback(residual):
    residuals.append(residual)

# example matrix and vector
A = np.array([
    [4, 12, -8],
    [12, 37, -19],
    [-8, -19, 50]
], dtype=float)

b = np.array([12, 59, 127], dtype=float)

# run GMRES
x = run_gmres(A, b)

# print the result
print("Solution x:", x)

# plot the residual curve
plt.figure()
plt.semilogy(residuals, marker='o')
plt.title('Convergence of GMRES')
plt.xlabel('Iteration')
plt.ylabel('Relative Residual (log scale)')
plt.grid(True)
plt.savefig('./203B_hw/figure/1-4.png')
plt.show()