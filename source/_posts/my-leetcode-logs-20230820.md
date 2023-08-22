---
title: my-leetcode-logs-20230820
date: 2023-08-20 18:05:18
tags:
- LeetCode
- 回溯算法章节
- C++
- Alibaba
---

## 131. 分割回文串（C++回溯法实现）

```
class Solution {
public:
    vector<string> path;
    vector<vector<string>> result;

    //判断字符串是否为回文串的函数
    bool isHuiWen(const string& s, int start, int end){
        for(int i = start, j = end; i < j;i++,j--){
            if(s[i] != s[j]){
                return false;
            }
        }
        return true;
    }

    void backtracing(const string& s, int startIndex){
        if(startIndex >= s.size()){
            result.push_back(path);
            return;
        }

        //单层回溯逻辑
        for(int i = startIndex; i < s.size();i++){
            if(isHuiWen(s, startIndex, i)){
                //截取字符串
                string tmp = s.substr(startIndex, i - startIndex + 1);
                path.push_back(tmp);
            }else{//如果不是回文数
                continue;
            }

            //递归调用
            backtracing(s, i + 1);
            path.pop_back();//回溯过程，返回已经添加的子串
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

## （复习）93. 复原 IP 地址（C++回溯实现）

```
class Solution {
public:
    vector<string> result;

    //判断是否合法的函数
    bool isValid(const string& s, int start, int end){
        if(start > end){
            return false;
        }

        //判断start对应的s中的字符是否为0
        if(s[start] == '0' && start != end){
            return false;
        }

        //然后进行判断
        int num = 0;
        for(int i = start; i <= end;i++){
            //首先判断每个数字是否合法
            if(s[i] > '9' && s[i] < '0'){
                return false;
            }

            //如果数组都合法，然后计算从start到end对应的数字之和是否超过了255
            num = num * 10 + (s[i] - '0');
            if(num > 255){
                return false;
            }
        }

        return true;
    }


    void backtracing(string& s, int startIndex, int pointNum){
        //终止条件
        if(pointNum == 3){
            //判断最后一个逗点之后的字符串是否合法
            if(isValid(s, startIndex, s.size() - 1)){
                result.push_back(s);
            }
            return;
        }

        //单层回溯逻辑
        for(int i = startIndex; i < s.size();i++){
            //判断是否合法
            if(isValid(s, startIndex, i)){
                s.insert(s.begin() + i + 1, '.');
                pointNum += 1;
                //递归调用函数，由于加入了一个点，因此下一个开始index应该为i + 2
                backtracing(s, i + 2, pointNum);
                pointNum -= 1;
                s.erase(s.begin() + i + 1);
            }else{
                break;
            }
        }
    }

    vector<string> restoreIpAddresses(string s) {
        result.clear();
        backtracing(s, 0, 0);
        return result;
    }
};
```

## （复习）78. 子集（C++实现）

```
class Solution {
public:

    vector<int> path;
    vector<vector<int>> result;
    
    void backtraing(vector<int>& nums, int startIndex){
        result.push_back(path);
        if(startIndex >= nums.size()){
            return;
        }

        //单层回溯逻辑
        for(int i = startIndex;i < nums.size();i++){
            path.push_back(nums[i]);
            backtraing(nums, i + 1);
            path.pop_back();
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        result.clear();
        path.clear();
        backtraing(nums, 0);
        return result;
    }
};
```
