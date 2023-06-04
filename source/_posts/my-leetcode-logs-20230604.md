---
title: my-leetcode-logs-20230604
date: 2023-06-04 16:52:42
tags:
- LeetCode
- Java
- alibaba
- 双指针
categories:
- LeetCode Logs
---

# 双指针相关题目
## 27.移除元素
```
class Solution {
    public int removeElement(int[] nums, int val) {
        int slow = 0;
        int fast = 0;

        while(fast < nums.length){
            if(nums[fast] != val){
                nums[slow++] = nums[fast];
            }
            fast++;
        }
        return slow;
    }
}
```