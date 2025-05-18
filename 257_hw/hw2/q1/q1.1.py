import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f2(x1, x2):
    return x1**2 + x2**2 + np.sin(10*x1) * np.sin(10*x2)


x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-2, 2, 400)
x1, x2 = np.meshgrid(x1, x2)


z = f2(x1, x2)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f2(x)')

plt.savefig('/root/code/257_hw/hw2/q1.png')
plt.show()