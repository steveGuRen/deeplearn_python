import numpy as np
#1, 创建一个 1 到 10 的一维数组
a = np.arange(1, 11)  # 从1到11（不包含11）
print(a)

#2,创建一个 3x3 的单位矩阵（对角线为 1，其它为 0）
identity_matrix = np.eye(3)
print(identity_matrix)

#3,生成一个 5x5 的随机矩阵，并找出最大值和最小值
# 生成 5x5 的随机矩阵，取值范围是 [0.0, 1.0)
random_matrix = np.random.rand(5, 5)
#找最大值
max_value = np.max(random_matrix)
# 找最小值
min_value = np.min(random_matrix)
# 打印结果
print("随机矩阵：")
print(random_matrix)
print("最大值：", max_value)
print("最小值：", min_value)

#4,对一个数组中的所有元素乘以 3
# 创建一个数组
arr = np.array([1, 2, 3, 4, 5])

# 所有元素乘以 3
result = arr * 3

# 打印结果
print("原数组：", arr)
print("乘以 3 后的数组：", result)

#5,将一个一维数组 reshape 成 3x4 的二维数组

# 创建一维数组
a = np.arange(12)  # [0, 1, 2, ..., 11]

# reshape 成 3x4 的二维数组
b = a.reshape(3, 4)
# 打印结果
print(b)

#6, ✅ 练习 6：计算一个二维数组的行平均值（每一行一个值）

a = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# 计算每一行的平均值
# axis=0 是“沿着竖直轴”（列的方向）
# axis=1 是“沿着水平轴”（行的方向）
row_means = np.mean(a, axis=1)

print(row_means)

#7, ✅ 创建一个 10x10 的矩阵，边界全为 1，内部全为 0
# 1. 创建一个 10x10 的全1矩阵
a = np.ones((10, 10), dtype=int)

# 2. 将内部区域（去掉边界的部分）赋值为0
# 取行索引从 1 到 len-2 的所有行；
#
# 取列索引从 1 到 len-2 的所有列；
#
# 也就是排除了第一行和最后一行，第一列和最后一列的矩阵“内部区域”。
a[1:-1, 1:-1] = 0

# 3. 打印结果
print(a)
