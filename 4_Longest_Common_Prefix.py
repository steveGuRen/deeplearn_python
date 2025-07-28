from typing import List

class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""

        # 以第一个字符串为初始前缀
        prefix = strs[0]

        for s in strs[1:]:
            # 不断缩短 prefix，直到 s 以 prefix 开头
            while not s.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""

        return prefix

sol = Solution()
print("答案是：" + sol.longestCommonPrefix(["flower", "flow", "flight"]))  # 输出: "fl"
print("答案是：" + sol.longestCommonPrefix(["dog", "racecar", "car"]))     # 输出: ""
