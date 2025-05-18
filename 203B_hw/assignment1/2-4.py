import numpy as np
import matplotlib.pyplot as plt

# 自定义不可分数据点
X1 = np.array([[1, 1], [1, 2], [2, 1], [2, 1.5]])  # 4个点属于类1
X2 = np.array([[1.5, 1.5], [1.5, 2.5], [2.5, 1.5]])  # 3个点属于类-1
X = np.vstack((X1, X2))
y = np.array([1] * 4 + [-1] * 3)

# 绘制数据
plt.scatter(X1[:, 0], X1[:, 1], color='b', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], color='r', label='Class -1')
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Infeasible SVM Example: Custom Non-linearly Separable Data')
plt.legend()
plt.savefig('./203B_hw/figure/2-3.png')
plt.show()