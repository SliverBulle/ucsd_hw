import numpy as np
import matplotlib.pyplot as plt

# define drop-wave function
def drop_wave(x1, x2):
    return - (1 + np.cos(12 * np.sqrt(x1**2 + x2**2))) / (0.5 * (x1**2 + x2**2) + 2)

# create grid
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z3 = drop_wave(X1, X2)

# plot contour
plt.figure(figsize=(8, 6))
contour3 = plt.contour(X1, X2, Z3, levels=[-0.5, -0.3, -0.1], colors=['blue', 'green', 'red'])
plt.clabel(contour3, inline=True, fontsize=8)
plt.title('Level Sets of Drop-wave Function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.savefig('./257_hw/figure/q1.3.png')
plt.show()