def two_sum(nums, target):
    lookup = {}  # 用于存储数字及其下标，类似java的map,key是数字，value是下标
    for i, num in enumerate(nums): # enumerate(nums) 是 Python 的一个函数，它会将列表 nums 同时按顺序提供：索引和元素。
        complement = target - num
        if complement in lookup:
            return [lookup[complement], i]
        lookup[num] = i # 把数字和下标存入字典中
    return []

# 示例 1
print(two_sum([2, 7, 11, 15], 9))  # 输出: [0, 1]

# 示例 2
print(two_sum([3, 2, 4], 6))  # 输出: [1, 2]

# 示例 3
print(two_sum([3, 3], 6))  # 输出: [0, 1]