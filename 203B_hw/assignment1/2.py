import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 生成线性可分的数据
np.random.seed(42)
X1 = np.random.randn(3, 2) + [2, 2]  # 3个点属于类1
X2 = np.random.randn(4, 2) + [-2, -2]  # 4个点属于类-1
X = np.vstack((X1, X2))
y = np.array([1] * 3 + [-1] * 4)

# 拟合SVM
svm = SVC(kernel='linear', C=1e5)
svm.fit(X, y)

# 获取分隔超平面
w = svm.coef_[0]
b = svm.intercept_[0]
xx = np.linspace(-4, 4)
yy = -(w[0] * xx + b) / w[1]

# 绘图
plt.scatter(X1[:, 0], X1[:, 1], color='b', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='r', label='Class -1')
plt.plot(xx, yy, 'k-', label='Hyperplane')
plt.fill_between(xx, yy - 1/svm.coef_[0][1], yy + 1/svm.coef_[0][1], color='gray', alpha=0.2, label='Margin')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('SVM with Linearly Separable Data')
plt.legend()
plt.show()
plt.savefig('./203B_hw/figure/2.png')