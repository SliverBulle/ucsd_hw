import numpy as np

# 示例矩阵 A
A = np.array([
    [4, 12, -8],
    [12, 37, -19],
    [-8, -19, 50]
], dtype=float)

# 计算奇异值
U, s, Vt = np.linalg.svd(A)

# 计算条件数（不涉及开根号）
condition_number_squared = (s[0] / s[-1])**2
print("Condition number squared:", condition_number_squared)

print(s[0])
print(s[-1])