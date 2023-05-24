---
title: my-leetcode-logs
date: 2023-05-24 15:03:57
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
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

## 11.盛最多水的容器

```
class Solution {
    public int maxArea(int[] height) {
        int n = height.length;
        int left = 0;
        int right = n - 1;

        int area = 0;
        while(left < right){
            int h = Math.min(height[left], height[right]);
            area = Math.max(area, h * (right - left));
            if(height[left] < height[right]){
                left ++;
            }else{
                right -- ;
            }
        }
        return area;
    }
}
```

## 15.三数之和

```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        //首先先排序（升序）
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;

        for(int i = 0; i < n; i++){
            if(nums[i] > 0){
                return result;
            }

            //判断i位置的元素是否和其前一个元素相同，相同那么进入下一次循环
            if(i > 0 && nums[i] == nums[i - 1]){
                continue;
            }

            int left = i + 1;
            int right = n - 1;

            while(left < right){
                if(nums[i] + nums[left] + nums[right] > 0){
                    right--;
                }else if(nums[i] + nums[left] + nums[right] < 0){
                    left++;
                }else{
                    List<Integer> tmp = new ArrayList<>();
                    tmp.add(nums[i]);
                    tmp.add(nums[left]);
                    tmp.add(nums[right]);
                    result.add(tmp);

                    while(right > left && nums[right - 1] == nums[right]){
                        right--;
                    }

                    while(right > left && nums[left + 1] == nums[left]){
                        left++;
                    }

                    right--;
                    left++;

                }
            }
        }
        return result;
    }
}
```
