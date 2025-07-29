class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        # 括号的映射关系
        mapping = {')': '(', ']': '[', '}': '{'}

        for char in s:
            if char in mapping:
                # 如果栈为空，给个默认值（防止空栈弹出报错）,下面的语法是A if 条件 else B
                top_element = stack.pop() if stack else '#'
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)

        # 如果栈为空，说明全部匹配成功
        return not stack


def test_isValid():
    solution = Solution()

    test_cases = [
        ("()", True),
        ("()[]{}", True),
        ("(]", False),
        ("([)]", False),
        ("([])", True),
        ("", True),  # 空字符串也视为有效
        ("(", False),
        ("{[()]}", True),
        ("{[(])}", False),
        ("(((((())))))", True),
        ("[", False),
        ("([]{})", True),
        ("))))", False),
        ("([)", False),
    ]

    for i, (s, expected) in enumerate(test_cases):
        result = solution.isValid(s)
        assert result == expected, f"Test case {i + 1} failed: input({s}) → expected {expected}, got {result}"

    print("All test cases passed!")


# 调用测试函数
test_isValid()
