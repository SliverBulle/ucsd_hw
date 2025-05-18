import numpy as np
import matplotlib.pyplot as plt

# define gaussian density function
def gaussian_density(x1, x2, mu, sigma):
    x = np.stack((x1, x2), axis=-1)
    sigma_inv = np.linalg.inv(sigma)
    diff = x - mu
    exponent = -0.5 * np.einsum('...i,ij,...j->...', diff, sigma_inv, diff)
    return np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))

# parameters
mu = np.array([1, 1])
sigma = np.array([[1, 1], [1, 2]])

# create grid
x1 = np.linspace(-3, 5, 400)
x2 = np.linspace(-3, 5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z2 = gaussian_density(X1, X2, mu, sigma)

# plot contour
plt.figure(figsize=(8, 6))
contour2 = plt.contour(X1, X2, Z2, levels=[0.05, 0.1, 0.15], colors=['blue', 'green', 'red'])
plt.clabel(contour2, inline=True, fontsize=8)
plt.title('Level Sets of Gaussian Density Function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.savefig('./257_hw/figure/q1.2.png')
plt.show()