
import numpy as np
import matplotlib.pyplot as plt

# define objective function
def f(x):
    return x[0]**2 - x[0]*x[1] + 3*x[1]**2 + 5

# define gradient
def grad_f(x):
    return np.array([2*x[0] - x[1], -x[0] + 6*x[1]])

# gradient descent
def gradient_descent(f, grad_f, x_init, alpha, max_iter=10):
    x = x_init
    points = [x.copy()]
    for _ in range(max_iter):
        grad = grad_f(x)
        x = x - alpha * grad
        points.append(x.copy())
    return points

# define Hessian matrix
def hessian_f(x):
    return np.array([[2, -1], [-1, 6]])
x_init = np.array([2.0, 2.0])
# newton method
def newton_descent(f, grad_f, hessian_f, x_init, alpha=1, max_iter=10):
    x = x_init
    points = [x.copy()]
    for _ in range(max_iter):
        grad = grad_f(x)
        hess = hessian_f(x)
        x = x - alpha * np.linalg.inv(hess).dot(grad)
        points.append(x.copy())
    return points

points_newton = newton_descent(f, grad_f, hessian_f, x_init)

# plot newton method points sequence
plt.figure(figsize=(8, 6))
points_newton = np.array(points_newton)
plt.plot(points_newton[:, 0], points_newton[:, 1], marker='s', label='Newton Descent (α=1)')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Newton Descent')
plt.grid(True)
plt.legend()
plt.savefig('./257_hw/figure/q5.3.png')
plt.show()