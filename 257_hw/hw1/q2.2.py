import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def target_function(x):
    return x[0]**2 + 4*x[1]**2

# 三点法
def three_point_method(f, x_init, epsilon=0.01, max_iter=30):
    x = x_init
    values = []
    for _ in range(max_iter):
        # record the current function value
        values.append(f(x))
        
        # generate a random vector p
        p = np.random.randn(*x.shape)
        p = epsilon * p / np.linalg.norm(p)
        
        # calculate the function value at three points
        f_x = f(x)
        f_xp = f(x + p)
        f_xm = f(x - p)
        
        # update x
        if f_xp < f_x and f_xp < f_xm:
            x_new = x + p
        elif f_xm < f_x:
            x_new = x - p
        else:
            x_new = x
        
        x = x_new
    
    return values

# gradient descent method
def gradient_descent(f, grad_f, x_init, epsilon=0.1, max_iter=30):
    x = x_init
    values = []
    for _ in range(max_iter):
        # record the current function value
        values.append(f(x))
        
        # calculate the gradient and normalize it
        grad = grad_f(x)
        grad = epsilon * grad / np.linalg.norm(grad)
        
        # update x
        x = x - grad
    
    return values

# define the gradient
def gradient(x):
    return np.array([2*x[0], 8*x[1]])

# initial point
x_init = np.array([1.0, 1.0])

# calculate the function value change
three_point_values = three_point_method(target_function, x_init)
gradient_descent_values = gradient_descent(target_function, gradient, x_init)

# plot the figure
plt.figure(figsize=(10, 6))
plt.plot(three_point_values, label='Three-Point Method', marker='o')
plt.plot(gradient_descent_values, label='Gradient Descent', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Function Value vs. Iteration')
plt.legend()
plt.grid(True)
plt.savefig('./257_hw/figure/q2.2.png')
plt.show()