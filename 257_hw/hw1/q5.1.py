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

# initial point
x_init = np.array([2.0, 2.0])

# suitable step size
alpha_converge = 0.1
points_converge = gradient_descent(f, grad_f, x_init, alpha_converge)

# plot convergence points sequence
plt.figure(figsize=(8, 6))
points_converge = np.array(points_converge)
plt.plot(points_converge[:, 0], points_converge[:, 1], marker='o', label=f'Gradient Descent (α={alpha_converge})')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gradient Descent Convergence')
plt.grid(True)
plt.legend()
plt.savefig('./257_hw/figure/q5.1.png')
plt.show()