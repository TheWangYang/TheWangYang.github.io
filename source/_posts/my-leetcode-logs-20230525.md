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

## 904.水果成篮

```
//超时写法
class Solution {
    public int totalFruit(int[] fruits) {
        int n = fruits.length;
        int start = 0;
        int max_num = -1;
        for(int i = 0; i < n;i++){
            while(isAboveTwo(fruits, start, i) && start < i){
                start ++;
            }
            max_num = max_num < (i - start) + 1 ? (i - start) + 1 : max_num;
        }
        return max_num;
    }

    //判断从start到i之间是否有超过两种不同类型的水果
    boolean isAboveTwo(int[] array, int left, int right){
        Map<Integer, Integer> dict = new HashMap<>();
        for(int i = left; i <= right;i++){
            if(!dict.containsKey(array[i]) && dict.size() < 2){
                dict.put(array[i], 0);
            }else if(!dict.containsKey(array[i]) && dict.size() == 2){
                return true;
            }else if(dict.containsKey(array[i])){
                continue;
            }
        }
        return false;
    }
}

//滑动窗口写法
class Solution {
    public int totalFruit(int[] fruits) {
        int n = fruits.length;
        int start = 0;
        int max_num = -1;
        Map<Integer, Integer> dict = new HashMap<>();
        for(int i = 0; i < n;i++){
            //获得原来存在map中的对应的水果种类的水果树数量
            dict.put(fruits[i], dict.getOrDefault(fruits[i], 0) + 1);
            //然后判断当前窗口中是否存在超过两种水果
            while(dict.size() > 2){
                //设置start对应的位置水果种类对应的树木数量 - 1
                dict.put(fruits[start], dict.get(fruits[start]) - 1);
                //如果在当前滑动窗口中不存在对应的种类的树木（即树木数量为0）
                if(dict.get(fruits[start]) == 0){
                    //那么直接删除dict字典中对应的key
                    dict.remove(fruits[start]);
                }
                //然后滑动窗口左边start向左移动一位
                start ++;
            }
            //然后，更新最大值
            max_num = max_num < (i - start) + 1 ? (i - start) + 1 : max_num;
        }
        return max_num;
    }
}
```

## 76.最小覆盖子串

```
class Solution {
    public String minWindow(String s, String t) {
        int sn = s.length();
        int tn = t.length();

        int start = 0;
        int minLen = Integer.MAX_VALUE;
        String result = "";

        Map<Character, Integer> tDict = new HashMap<>();
        for (int i = 0; i < tn; i++) {
            tDict.put(t.charAt(i), tDict.getOrDefault(t.charAt(i), 0) + 1);
        }

        Map<Character, Integer> windowDict = new HashMap<>();
        int formed = 0; // 记录窗口中满足条件的字符数量

        int left = 0;
        int right = 0;

        while (right < sn) {
            char c = s.charAt(right);
            windowDict.put(c, windowDict.getOrDefault(c, 0) + 1);

            if (tDict.containsKey(c) && windowDict.get(c).intValue() == tDict.get(c).intValue()) {
                formed++;
            }

            while (left <= right && formed == tDict.size()) {
                // 更新最小窗口长度和结果
                int curLen = right - left + 1;
                if (curLen < minLen) {
                    minLen = curLen;
                    result = s.substring(left, right + 1);
                }

                // 缩小窗口左边界
                char leftChar = s.charAt(left);
                windowDict.put(leftChar, windowDict.get(leftChar) - 1);
                if (tDict.containsKey(leftChar) && windowDict.get(leftChar).intValue() < tDict.get(leftChar).intValue()) {
                    formed--;
                }

                left++;
            }

            right++;
        }

        return result;
    }

}
```

##
