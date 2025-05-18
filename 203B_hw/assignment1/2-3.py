import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Custom non-linearly separable data points
X1 = np.array([[1, 1], [1, 2], [2, 1], [2, 1.5]])  # 4 points for class 1
X2 = np.array([[1.5, 1.5], [1.5, 2.5], [2.5, 1.5]])  # 3 points for class -1
X = np.vstack((X1, X2))
y = np.array([1] * 4 + [-1] * 3)

# Fit Soft Margin SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# Get the separating hyperplane
w = svm.coef_[0]
b = svm.intercept_[0]
xx = np.linspace(0, 3)
yy = -(w[0] * xx + b) / w[1]

# Plot
plt.scatter(X1[:, 0], X1[:, 1], color='b', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='r', label='Class -1')
plt.plot(xx, yy, 'k-', label='Hyperplane')
plt.fill_between(xx, yy - 1/svm.coef_[0][1], yy + 1/svm.coef_[0][1], color='gray', alpha=0.2, label='Margin')
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Soft Margin SVM with Non-linearly Separable Data')
plt.legend()
plt.savefig('./203B_hw/figure/2-4.png')
plt.show()