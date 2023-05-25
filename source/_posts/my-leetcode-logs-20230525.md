---
title: my-leetcode-logs-20230525
date: 2023-05-25 19:37:23
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---

## 209.长度最小的子数组

```
//滑动窗口
class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int result = 1000000001;
        int start = 0;
        int sum = 0;
        for(int i = 0; i < nums.length;i++){
            sum += nums[i];
            while(sum >= target){
                int in_result = (i - start) + 1;
                result = result < in_result ? result : in_result;
                sum -= nums[start++];
            }
        }
        return result == 1000000001 ? 0 : result;
    }
}
```

##
