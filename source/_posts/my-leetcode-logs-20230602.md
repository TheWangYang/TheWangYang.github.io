---
title: my-leetcode-logs-20230602
date: 2023-06-02 21:42:07
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---

## 459.重复的子字符串（需要不定时回顾，使用了KMP算法）
```
class Solution {
    //得到前缀表的函数
    void getNext(int[] next, String s){
        next[0] = 0;
        int j = 0;
        for(int i = 1;i < s.length();i++){
            while(j > 0 && s.charAt(i) != s.charAt(j)){
                j = next[j - 1];
            }
            //判断字符串s中对应位置为i和j是否包含相等的字符
            if(s.charAt(i) == s.charAt(j)){
                j++;
            }
            next[i] = j;
        }
    }

    public boolean repeatedSubstringPattern(String s) {
        //使用KMP字符串匹配算法实现
        if(s.length() == 0){
            return false;
        }

        //构造长度为s.length()的next数组
        int[] next = new int[s.length()];
        getNext(next, s);
        int len = s.length();
        if(next[len - 1] != 0 && len % (len - next[len - 1]) == 0){
            return true;
        }
        return false;
    }
}
```

## 
