import numpy as np
import matplotlib.pyplot as plt

# define function f1
def f1(x1, x2):
    return x1**2 + 2*x1*x2 + 2*x2**2

# create grid
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z1 = f1(X1, X2)

# plot contour
plt.figure(figsize=(8, 6))
contour1 = plt.contour(X1, X2, Z1, levels=[5, 10, 20], colors=['blue', 'green', 'red'])
plt.clabel(contour1, inline=True, fontsize=8)
plt.title('Level Sets of $f_1(x_1, x_2)$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.savefig('./257_hw/figure/q1.1.png')
plt.show()
