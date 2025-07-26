def is_palindrome(x):
    # 负数一定不是回文，比如 -121
    # 以0结尾的数字（除了0本身）也不是回文，例如10 -> 01
    if x < 0 or (x % 10 == 0 and x != 0):
        return False

    reverted = 0
    while x > reverted: # 当x小于reverted时，说明已经处理了一半的数字
        # 取出最后一位，加到reverted后面
        reverted = reverted * 10 + x % 10
        x //= 10  # 去掉最后一位

    # 当数字位数为偶数时，x == reverted
    # 当数字位数为奇数时，reverted多一位，需要去掉中间那一位再比较
    return x == reverted or x == reverted // 10

print(is_palindrome(1221))
print(is_palindrome(121))
print(is_palindrome(-121))
print(is_palindrome(10))
print(is_palindrome(12345))