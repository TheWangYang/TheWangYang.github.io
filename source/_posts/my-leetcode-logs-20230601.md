---
title: my-leetcode-logs-20230601
date: 2023-06-01 16:08:41
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---


## 350.两个数组的交集
```
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        int[] record1 = new int[1001];
        int[] record2 = new int[1001];

        int nums1Length = nums1.length;
        int nums2Length = nums2.length;

        for(int i = 0; i < nums1Length;i++){
            record1[nums1[i]]++; 
        }

        for(int i = 0; i< nums2Length;i++){
            record2[nums2[i]]++;
        }

        List<Integer> tmp = new ArrayList<Integer>();
        
        int n = Math.max(record1.length, record2.length);
        
        for(int i = 0;i < n;i++){
            if(record1[i] > 0 && record2[i] > 0){
                int m = Math.min(record1[i], record2[i]);
                for(int j = 0;j < m;j++){
                    tmp.add(i);
                }
            }
        }

        int[] result = new int[tmp.size()];
        for(int i = 0; i < tmp.size();i++){
            result[i] = tmp.get(i);
        }
        return result; 
    }
}
```

## 202.快乐数
```
class Solution {
    //得到数字n的每个位置上的数字平方和
    int getSum(int n){
        int sum = 0;
        while(n != 0){
            sum += (n % 10) * (n % 10);
            n = n / 10;
        }
        return sum;
    }

    public boolean isHappy(int n) {
        Map<Integer, Integer> dict = new HashMap<>();

        while(true){
            int currSum = getSum(n);
            if(currSum == 1){
                return true;
            }

            if(dict.containsKey(currSum) && dict.get(currSum) != 0){
                return false;
            }else{
                dict.put(currSum, 1);
            }
            n = currSum;
        }
    }
}
```

## 454.四数相加II
```
class Solution {
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer,Integer> map = new HashMap<>();

        for(int i = 0;i < nums1.length;i++){
            for(int j = 0; j < nums2.length;j++){
                map.put(nums1[i] + nums2[j], map.getOrDefault(nums1[i] + nums2[j], 0) + 1);
            }
        }

        int count = 0;
        for(int i = 0; i < nums3.length;i++){
            for(int j = 0;j < nums4.length;j++){
                count += map.getOrDefault(0 - (nums3[i] + nums4[j]), 0);
            }
        }
        return count;
    }
}
```

## 
```

```