---
title: my-leetcode-logs-20230728
date: 2023-07-28 12:40:08
tags:
- LeetCode
- 回溯算法章节
- C++
- Alibaba
categories:
- LeetCode Logs
---

## 复习：77. 组合（C++回溯+剪枝实现）

```
class Solution {
public:

    //定义保存路径的数组
    vector<int> path;
    //定义保存整个结果的result数组
    vector<vector<int>> result;
    //回溯实现
    void huisu(int n, int k, int startIndex){
        if(path.size() == k){
            result.push_back(path);
            return ;
        }

        for(int i = startIndex; i <= n - (k - path.size()) + 1; i++){
            path.push_back(i);
            huisu(n, k, i + 1);
            path.pop_back();
        }
    }

    vector<vector<int>> combine(int n, int k) {
        result.clear();
        huisu(n, k, 1);
        return result;
    }
};
```

## 216. 组合总和 III（C++回溯实现）

```
class Solution {
public:

    vector<int> path;
    vector<vector<int>> result;

    //回溯算法
    void backtracing(int k, int n, int startIndex, int sum){
        //判断path.size()是否等于k
        if(path.size() == k){
            //判断targetSum是否等于n
            if(sum == n){
                result.push_back(path);
            }
            return;
        }

        //回溯单层逻辑
        for(int i = startIndex; i <= 9;i++){
            path.push_back(i);
            sum += i;
            backtracing(k, n, i + 1, sum);
            sum -= i;
            path.pop_back();
        }

        return;
    }

    vector<vector<int>> combinationSum3(int k, int n) {
        result.clear();
        backtracing(k, n, 1, 0);
        return result;
    }
};
```

## 17. 电话号码的字母组合（C++回溯实现）

```
class Solution {
public:
    //定义字母和数字之间的映射
    const string num2str[10] = {
        "",//0
        "",//1
        "abc",//2
        "def",//3
        "ghi",//4
        "jkl",//5
        "mno",//6
        "pqrs",//7
        "tuv",//8
        "wxyz",//9
    };

    vector<string> result;
    string s;//这个string类型变量就相当于path

    //回溯算法实现
    //参数为传入digits地址，起始index下标
    void backtracing(const string& digits, int startIndex){
        if(startIndex == digits.size()){
            result.push_back(s);
            return;
        }
        
        //将digits中对应startIndex位置的数字转为数字
        int digit = digits[startIndex] - '0';
        string letters = num2str[digit];

        //单层回溯逻辑
        for(int i = 0;i < letters.size();i++){
            s.push_back(letters[i]);
            backtracing(digits, startIndex + 1);
            s.pop_back();
        }

        return ;
    }


    vector<string> letterCombinations(string digits) {
        s.clear();
        result.clear();
        if(digits.size() == 0){
            return result;
        }

        backtracing(digits, 0);
        return result;
    }
};
```

## 39. 组合总和（C++回溯法实现）

```
class Solution {
public:
    vector<int> path;
    vector<vector<int>> result;

    //回溯法实现
    void backtracing(vector<int>& candidates, int target, int sum, int index){
         if (sum > target) {
            return;
        }
        
        if(sum == target){
            result.push_back(path);
            return ;
        }

        for(int i = index; i < candidates.size();i++){
            path.push_back(candidates[i]);
            sum += candidates[i];
            backtracing(candidates, target, sum, i);
            sum -= candidates[i];
            path.pop_back();
        }
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        result.clear();
        path.clear();

        backtracing(candidates, target, 0, 0);

        return result;
    }
};
```


