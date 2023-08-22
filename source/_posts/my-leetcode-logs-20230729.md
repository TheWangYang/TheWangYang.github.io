---
title: my-leetcode-logs-20230729
date: 2023-07-29 15:01:37
tags:
- LeetCode
- 回溯算法章节
- C++
- Alibaba
categories:
- LeetCode Logs
---


## （复习）39. 组合总和（C++回溯法实现）

```
class Solution {
public:
    vector<int> path;
    vector<vector<int>> result;

    //回溯算法
    void backtracing(vector<int>& candidates, int target, int sum, int index){
        if(sum > target){
            return ;
        }

        if(sum == target){
            result.push_back(path);
            return ;
        }

        for(int i = index; i < candidates.size(); i++){
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

## 40. 组合总和 II（C++回溯法实现）

```
class Solution {
public:
    //核心思想：需要增加used数据，来判断是否使用过
    vector<int> path;
    vector<vector<int>> result;

    void backtracing(vector<int>& candidates, int target, int sum, int index, vector<bool>& used){
        if(sum > target){
            return ;
        }

        if(sum == target){
            result.push_back(path);
            return ;
        }

        for(int i = index; i < candidates.size() && sum + candidates[i] <= target;i++){
            //判断used数组中是否已经使用过i位置元素
            if(i > 0 && candidates[i] == candidates[i - 1] && used[i - 1] == false){
                continue;
            }

            path.push_back(candidates[i]);
            sum += candidates[i];
            used[i] = true;
            backtracing(candidates, target, sum, i + 1, used);
            used[i] = false;
            sum -= candidates[i];
            path.pop_back();
        }
    }

    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        vector<bool> used(candidates.size(), false);
        result.clear();
        path.clear();

        //先将candidates进行排序
        sort(candidates.begin(), candidates.end());

        backtracing(candidates, target, 0, 0, used);

        return result;
    }
};
```

## 131. 分割回文串（C++回溯法实现，无优化版）

```
class Solution {
public:
    //判断是否为回文字符串的函数
    bool isHuiWen(const string& huiwen, int start, int end){
        for(int i = start, j = end; i < j; i++, j--){
            if(huiwen[i] != huiwen[j]){
                return false;
            }
        }
        return true;
    }

    //回溯法实现
    vector<string> path;
    vector<vector<string>> result;

    void backtracing(string s, int startIndex){
        if(startIndex >= s.size()){
            result.push_back(path);
            return ;
        }

        //回溯法中单层逻辑
        for(int i = startIndex; i < s.size(); i ++){
            if(isHuiWen(s, startIndex, i)){
                string tmp = s.substr(startIndex, i - startIndex + 1);
                path.push_back(tmp);
            }else{
                continue;
            }

            //回溯调用
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
