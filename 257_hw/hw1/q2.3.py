import numpy as np
import matplotlib.pyplot as plt

# define drop-wave function
def drop_wave(x):
    return - (1 + np.cos(12 * np.sqrt(x[0]**2 + x[1]**2))) / (0.5 * (x[0]**2 + x[1]**2) + 2)

# three-point method
def three_point_method(f, x_init, epsilon=0.1, max_iter=30):
    x = x_init
    values = []
    for _ in range(max_iter):
        # record current function value
        values.append(f(x))
        
        # generate random vector p
        p = np.random.randn(*x.shape)
        p = epsilon * p / np.linalg.norm(p)
        
        # calculate function value of three points
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

# gradient descent
def gradient_descent(f, grad_f, x_init, epsilon=0.1, max_iter=30):
    x = x_init
    values = []
    for _ in range(max_iter):
        # record current function value
        values.append(f(x))
        
        # calculate gradient and normalize
        grad = grad_f(x)
        grad = epsilon * grad / np.linalg.norm(grad)
        
        # 更新 x
        x = x - grad
    
    return values

# define numerical gradient of drop-wave function
def numerical_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_h1 = np.array(x, dtype=float)
        x_h2 = np.array(x, dtype=float)
        x_h1[i] += h
        x_h2[i] -= h
        grad[i] = (f(x_h1) - f(x_h2)) / (2 * h)
    return grad

# 初始点
x_init = np.array([2.0, 2.0])

# calculate function value change
three_point_values = three_point_method(drop_wave, x_init)
gradient_descent_values = gradient_descent(drop_wave, lambda x: numerical_gradient(drop_wave, x), x_init)

# plot figure
plt.figure(figsize=(10, 6))
plt.plot(three_point_values, label='Three-Point Method', marker='o')
plt.plot(gradient_descent_values, label='Gradient Descent', marker='x')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Function Value vs. Iteration for Drop-wave Function')
plt.legend()
plt.grid(True)
plt.savefig('./257_hw/figure/q2.3.png')
plt.show()