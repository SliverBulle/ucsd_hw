import cvxpy as cp
import numpy as np

# 定义变量
x = cp.Variable(4)

# 定义原始问题的参数
A = np.array([
    [1, 2, 3, 4],
    [0, 1, 0, 1],
    [2, 0, 1, 0],
    [1, 1, 1, 1],
    [0, 0, 1, 1]
])
b = np.array([4, 3, 4, 1, 2])
c = np.array([-1, -2, -3, -4])

# 定义原始问题
objective = cp.Minimize(c @ x)
constraints = [A @ x <= b, x >= 0]
prob = cp.Problem(objective, constraints)

# 求解原始问题
prob.solve(solver=cp.SCS, verbose=True)
print("Primal problem solution:")
print("Optimal value:", prob.value)
print("Optimal solution:", x.value)

y = cp.Variable(5)
# 定义对偶问题
objective_dual = cp.Maximize(b @ y)
constraints_dual = [A.T @ y >= c, y >= 0]
prob_dual = cp.Problem(objective_dual, constraints_dual)
prob_dual.solve(solver=cp.SCS, verbose=True)

print("Dual problem solution:")
print("Optimal value (dual):", prob_dual.value)
print("Optimal solution (dual):", y.value)

