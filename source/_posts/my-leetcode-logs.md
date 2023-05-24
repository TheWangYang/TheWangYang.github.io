---
title: my-leetcode-logs
date: 2023-05-24 15:03:57
tags:
- LeetCode
- Java
- alibaba
---


# My LeetCode HOT 100 logs

*use language: java*

## 1.两数之和

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[0];
    }
}
```

## 49.字母异位词分组

```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for(String str: strs){
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

## 128.最长连续序列

```
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for(int num : nums){
            set.add(num);
        }

        int result = 0;

        for(int num : nums){
            if(!set.contains(num - 1)){
                int currNum = num;
                int inner_result = 1;

                while(set.contains(currNum + 1)){
                    inner_result += 1;
                    currNum += 1;
                }
                
                result = Math.max(result, inner_result);
            }
        }
        return result;
    }
}

```

## 283.移动零

```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        
        int lp = 0;
        int rp = lp;

        while(lp != n - 1){
            if(nums[lp] == 0){
                while(rp != n){
                    if(nums[rp] != 0){
                        int tmp = nums[rp];
                        nums[rp] = nums[lp];
                        nums[lp] = tmp;
                        break;
                    }
                    rp += 1;
                }
            }
            lp += 1;
            rp = lp;
        }
    }
}
```
