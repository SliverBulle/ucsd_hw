from scipy.optimize import linprog
import numpy as np
# 定义矩阵 A 和向量 b
A = np.array([
    [9, 3, -1, 2, 0, 1, 0],
    [-3, 1, 1, 1, 1, 1, 1],
    [5, 0, 1, 1, 0, 0, -1],
    [1, 0, 1, 0, 1, 0, 0],
    [1, 0, 2, 0, -1, -1, 0]
])
b = np.array([1, 5, 2, 3, 2])

# 定义目标函数系数 c
c = np.array([1, 2, 3, 4, 5, 6, 7])

# 使用 scipy.optimize.linprog 求解
result = linprog(c, A_ub=A, b_ub=b,bounds=(None, None), method='highs')

print("2.1 Primal Solution:", result.x)
print("2.1 function value:", result.fun)
# 使用 scipy.optimize.linprog 求解
result = linprog(c, A_eq=A, b_eq=b,bounds=(None, None), method='highs')

print("2.2 Primal Solution:", result.x)
print("2.2 function value:", result.fun)
# 使用 scipy.optimize.linprog 求解
result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')

print("2.3 Primal Solution:", result.x)
print("2.3 function value:", result.fun) 
# 使用 scipy.optimize.linprog 求解
result = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method='highs')

print("2.4 Primal Solution:", result.x)
print("2.4 function value:", result.fun)

# 对偶问题的系数
A_dual = A.T
b_dual = c
c_dual = -b

# II.1.1. 对偶问题: maximize b^T y subject to A^T y = c, y >= 0
result_dual_1 = linprog(b, A_eq=A.T, b_eq=-c, bounds=(0,None))

# II.1.2. 对偶问题: maximize b^T y subject to A^T y = c
result_dual_2 = linprog(b, A_eq=A.T, b_eq=-c,bounds = (None,None))

# II.1.3. 对偶问题: maximize -b^T ν subject to A^T ν + c >= 0, ν >= 0
result_dual_3 = linprog(b, A_ub=-A.T, b_ub=c, bounds=(0,None))

# II.1.4. 对偶问题: maximize -b^T ν subject to A^T ν + c >= 0
result_dual_4 = linprog(b, A_ub=-A.T, b_ub=c,bounds = (None,None))

# 打印对偶问题的结果
print("Dual Result for II.1.1:", result_dual_1.fun)
print("v value:", result_dual_1.x)
print("Dual Result for II.1.2:", result_dual_2.fun)
print("v value:", result_dual_2.x)
print("Dual Result for II.1.3:", result_dual_3.fun)
print("v value:", result_dual_3.x)
print("Dual Result for II.1.4:", result_dual_4.fun)
print("v value:", result_dual_4.x)