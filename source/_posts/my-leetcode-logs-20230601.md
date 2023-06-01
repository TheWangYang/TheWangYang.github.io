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

## 