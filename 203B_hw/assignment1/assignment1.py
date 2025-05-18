import numpy as np
import matplotlib.pyplot as plt

def conjugate_gradient(A, b, tol=1e-8):
    x = np.zeros_like(b)  # Initial guess x^(0) = 0
    r = b - A @ x  # Initial residual r^(0)
    p = r.copy()  # Initial search direction p^(0)
    rsold = np.dot(r, r)  # Initial residual norm squared
    b_norm = np.linalg.norm(b)
    
    residuals = [np.sqrt(rsold) / b_norm]  # Store initial relative residual

    k = 0
    while residuals[-1] > tol:
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        residuals.append(np.sqrt(rsnew) / b_norm)
        
        if residuals[-1] < tol:
            break
        
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
        
        print(f"Iteration {k}: x = {x}, relative residual = {residuals[-1]}")
        k += 1

    return x, residuals

# Updated matrix A and vector b
A = np.array([
    [4, 12, -8],
    [12, 37, -19],
    [-8, -19, 50]
], dtype=float)

b = np.array([12, 59, 127], dtype=float)

x, residuals = conjugate_gradient(A, b)

# Plotting the relative residuals
plt.figure()
plt.semilogy(residuals, marker='o')
plt.title('Convergence of Conjugate Gradient')
plt.xlabel('Iteration')
plt.ylabel('Relative Residual (log scale)')
plt.grid(True)
plt.savefig('./203B_hw/figure/1-5-3.png')
plt.show()