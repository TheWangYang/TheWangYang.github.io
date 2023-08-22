---
title: my-leetcode-logs-20230802
date: 2023-08-02 13:59:48
tags:
- LeetCode
- 回溯算法章节
- C++
- Alibaba
categories:
- LeetCode Logs
---

## （复习）131. 分割回文串（C++回溯法实现，无优化版）

```
class Solution {
public:
    vector<string> path;
    vector<vector<string>> result;

    //设置函数判断输入字符串是否为回文串
    bool isHuiWen(const string& huiwen, int start, int end){
        for(int i = start, j = end; i < j;i++, j--){
            if(huiwen[i] != huiwen[j]){
                return false;
            }
        }
        return true;
    }

    //回溯法实现
    void backtracing(string s, int startIndex){
        if(startIndex >= s.size()){
            result.push_back(path);
            return ;
        }

        //回溯中单层逻辑实现
        for(int i = startIndex;i < s.size();i++){
            if(isHuiWen(s, startIndex, i)){
                string tmp = s.substr(startIndex, i - startIndex + 1);
                path.push_back(tmp);
            }else{
                continue;
            }
            //递归调用
            backtracing(s, i + 1);
            path.pop_back();
        }
    }

    vector<vector<string>> partition(string s) {
        result.clear();
        path.clear();
        backtracing(s, 0);
        return result;
    }
};
```

## 93. 复原 IP 地址（C++回溯实现）

```
class Solution {
public:
    vector<string> result;

    //判断是否为合法
    bool isValid(string s, int start, int end){
        if(start > end){
            return false;
        }

        //两个逗号之间的字符串的开始位置为0，表示不合法
        if(s[start] == '0' && start != end){
            return false;
        }

        //记录两个逗号之间的数字相加是否在255范围之内
        int num = 0;

        for(int i = start; i <= end;i ++){
            if(s[i] > '9' || s[i] < '0'){
                return false;
            }
            //num累加
            num = num * 10 + (s[i] - '0');
            
            //判断num是否大于255
            if(num > 255){
                return false;
            }
        }

        return true;
    }


    //设置的回溯算法
    void backtracing(string s, int startIndex, int pointSum){
        //判断pointSum等于3，表示是一个可能的结果
        if(pointSum == 3){
            if(isValid(s, startIndex, s.size() - 1)){
                result.push_back(s);
                return ;
            }
        }

        //单层回溯逻辑
        for(int i = startIndex; i < s.size(); i ++){
            //首先判断是否合法
            if(isValid(s, startIndex, i)){
                //在i位置之后插入一个点
                s.insert(s.begin() + i + 1, '.');
                pointSum += 1;
                //回溯调用
                backtracing(s, i + 2, pointSum);
                pointSum -= 1;
                s.erase(s.begin() + i + 1);
            }else{
                break;
            }
        }
        return ;
    }

    vector<string> restoreIpAddresses(string s) {
        result.clear();
        backtracing(s, 0, 0);
        return result;
    }
};
```

## 78. 子集（C++回溯法实现）

```
class Solution {
public:
    vector<vector<int>> result;
    vector<int> path;
    void backtracing(vector<int>& nums, int startIndex){
        result.push_back(path);
        if(startIndex > nums.size()){
            return;
        }

        for(int i = startIndex; i < nums.size();i++){
            path.push_back(nums[i]);
            backtracing(nums, i + 1);
            path.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        result.clear();
        path.clear();
        backtracing(nums, 0);
        return result;
    }
};
```

## 实现了
