import numpy as np
import time
from scipy.linalg import lu, solve_triangular
from scipy.sparse.linalg import gmres

def conjugate_gradient(A, b, tol=1e-8):
    x = np.zeros_like(b)
    r = b - A @ x
    p = r.copy()
    rsold = np.dot(r, r)
    b_norm = np.linalg.norm(b)
    
    residuals = [np.sqrt(rsold) / b_norm]

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

    return x

# 示例矩阵 A 和向量 b
A = np.array([
    [4, 12, -8],
    [12, 37, -19],
    [-8, -19, 50]
], dtype=float)

b = np.array([12, 59, 127], dtype=float)

# 测量CG的时间
start_time = time.time()
x_cg = conjugate_gradient(A, b)
end_time = time.time()
time_cg = end_time - start_time
print(f"CG time: {time_cg:.6f} seconds")

# 测量GMRES的时间
start_time = time.time()
x_gmres, exitCode = gmres(A, b, tol=1e-8)
end_time = time.time()
time_gmres = end_time - start_time
print(f"GMRES time: {time_gmres:.6f} seconds")

# 测量高斯消元的时间（使用numpy的solve）
start_time = time.time()
x_gaussian = np.linalg.solve(A, b)
end_time = time.time()
time_gaussian = end_time - start_time
print(f"Gaussian Elimination time: {time_gaussian:.6f} seconds")

# 测量LU分解的时间
start_time = time.time()
P, L, U = lu(A)
y = solve_triangular(L, b, lower=True)  # 使用 solve_triangular 处理下三角矩阵
x_lu = solve_triangular(U, y, lower=False)  # 使用 solve_triangular 处理上三角矩阵
end_time = time.time()
time_lu = end_time - start_time
print(f"LU Decomposition time: {time_lu:.6f} seconds")

# 比较所有方法的时间
times = {
    "CG": time_cg,
    "GMRES": time_gmres,
    "Gaussian Elimination": time_gaussian,
    "LU Decomposition": time_lu
}

fastest_method = min(times, key=times.get)
print(f"The fastest method is: {fastest_method}")