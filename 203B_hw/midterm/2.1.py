from sympy import Matrix

# 定义矩阵 A
A = Matrix([
    [9, 3, -1, 2, 0, 1, 0],
    [-3, 1, 1, 1, 1, 1, 1],
    [5, 0, 1, 1, 0, 0, -1],
    [1, 0, 1, 0, 1, 0, 0],
    [1, 0, 2, 0, -1, -1, 0]
])

# 定义向量 c
c = Matrix([1, 2, 3, 4, 5, 6, 7])
# 构造增广矩阵 [A^T | -c]
augmented_matrix = A.T.row_join(-c)

# 计算RREF
rref_matrix, pivots = augmented_matrix.rref()

# 输出结果
print("增广矩阵:")
print(augmented_matrix)
print("\n行简化阶梯形:")
print(rref_matrix)
print("\n主元列:", pivots)

