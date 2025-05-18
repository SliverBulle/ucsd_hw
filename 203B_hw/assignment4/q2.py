import cvxpy as cp
import numpy as np

# Define the matrices
A0 = np.array([[-300, 5, -5],
               [5, 4, -1],
               [-5, -1, 4]])

A1 = np.array([[-4, -5, 5],
               [-5, 4, 5],
               [5, 5, -4]])

A2 = np.array([[4, -5, -5],
               [-5, 2, -3],
               [-5, -3, 2]])

# Define variables
alpha = cp.Variable(3)  # [α0, α1, α2]
t = cp.Variable()

# Objective: Minimize t (maximum eigenvalue)
objective = cp.Minimize(t)

# Constraints
constraints = [
    alpha >= 0,
    cp.sum(alpha) == 1,
    t * np.eye(3) - (alpha[0]*A0 + alpha[1]*A1 + alpha[2]*A2) >> 0
]

# Solve the SDP
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.SCS, verbose=True)

# Output results
print(f"Optimal t (minimal maximum eigenvalue): {t.value:.4f}")
print(f"Optimal α: α0 = {alpha[0].value:.4f}, α1 = {alpha[1].value:.4f}, α2 = {alpha[2].value:.4f}")

# Verify by computing eigenvalues of F
F_opt = alpha[0].value * A0 + alpha[1].value * A1 + alpha[2].value * A2
eigenvalues = np.linalg.eigvalsh(F_opt)
print(f"Computed eigenvalues of F: {np.sort(eigenvalues)}")