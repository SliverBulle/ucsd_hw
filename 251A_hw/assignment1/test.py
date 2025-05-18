from sklearn.utils import resample
import numpy as np

# 创建一个示例数据集
data = np.array([1, 2, 3, 4, 5])

# 重采样数据集，生成一个新的子集
# 这里的 n_samples 指定了子集的大小
resampled_data = resample(data, n_samples=3, random_state=42)

print("Original data:", data)
print("Resampled data:", resampled_data)