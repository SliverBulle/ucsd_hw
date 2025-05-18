class BinaryIndexedTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
    
    def update(self, index, delta):
        """更新指定索引的值"""
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)
    
    def query(self, index):
        """查询从1到index的前缀和"""
        sum = 0
        while index > 0:
            sum += self.tree[index]
            index -= index & (-index)
        return sum
    
    def range_query(self, left, right):
        """查询区间[left, right]的和"""
        return self.query(right) - self.query(left - 1)

# 使用示例
def array_to_bit(arr):
    bit = BinaryIndexedTree(len(arr))
    for i in range(len(arr)):
        bit.update(i + 1, arr[i])
    return bit

def count_smaller_numbers(arr):
    n = len(arr)
    # 离散化数组
    sorted_arr = sorted(set(arr))
    rank = {val: i + 1 for i, val in enumerate(sorted_arr)}
    #print(rank)
    # 计算左边比当前数小的数量 (L_i)
    left_smaller = []
    bit_left = BinaryIndexedTree(len(rank))
    
    for num in arr:
        # 查询比当前数小的数的数量
        left_count = bit_left.query(rank[num] - 1)
        left_smaller.append(left_count)
        # 更新树状数组
        bit_left.update(rank[num], 1)
    
    # 计算右边比当前数小的数量 (R_i)
    right_smaller = []
    bit_right = BinaryIndexedTree(len(rank))
    
    for num in reversed(arr):
        # 查询比当前数小的数的数量
        right_count = bit_right.query(rank[num] - 1)
        right_smaller.append(right_count)
        # 更新树状数组
        bit_right.update(rank[num], 1)
    
    # 将右边的结果反转回正确顺序
    right_smaller.reverse()
    
    return left_smaller, right_smaller

# 使用示例
arr = [5, 2, 6, 1]
L, R = count_smaller_numbers(arr)
ans = 0
for i in range(len(arr)):
    ans += L[i] * R[i]

print(f"原数组: {arr}")
print(f"L_i (左边比当前数小的数量): {L}")
print(f"R_i (右边比当前数小的数量): {R}")