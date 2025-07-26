# 我们可以通过遍历罗马数字字符串 s，将其逐个字符转换为对应的数值，并根据“小值在大值前要减，小值在大值后要加”的规则进行累加，从而得到整数值。
def roman_to_int(s):
    #定义字典
    roman_map = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }

    total = 0  # 存储最终结果
    prev_value = 0  # 上一个字符的数值

    for ch in reversed(s):  # 反向遍历字符串
        value = roman_map[ch]
        if value < prev_value:
            total -= value  # 小于右边的值，减去它
        else:
            total += value  # 否则加上它
        prev_value = value  # 更新上一个字符的值

    return total

print(roman_to_int("MCMXCIV"))  # 输出: 1994
print(roman_to_int("III"))      # 输出: 3
print(roman_to_int("LVIII"))    # 输出: 58

