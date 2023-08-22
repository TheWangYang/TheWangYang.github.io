---
title: my-leetcode-logs-20230817
date: 2023-08-17 15:26:11
tags:
- LeetCode
- 回溯算法章节
- C++
- Alibaba
---


## （复习）40. 组合总和 II（C++回溯算法实现）

```
class Solution {
public:
    //定义两个数组
    vector<vector<int>> result;
    vector<int> path;

    //定义回溯函数
    void backtracing(vector<int>& candidates, int target, int sum, int startIndex,vector<bool>& used){
        if(sum ==target){
            result.push_back(path);
            return;
        }

        //单层循环逻辑
        for(int i = startIndex;i < candidates.size() && sum + candidates[i] <= target;i++){
            
            if(i > 0 && candidates[i] == candidates[i-1] && used[i - 1]==false){
                continue;//表示树的同层中的上一个一样的元素使用过了
            }

            sum += candidates[i];
            path.push_back(candidates[i]);
            used[i] = true;
            backtracing(candidates, target, sum, i + 1, used);
            used[i] = false;
            sum -= candidates[i];
            path.pop_back();
        }
    }


    vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
        //实现了函数中间
        result.clear();
        path.clear();
        vector<bool> used(candidates.size(), false);
        sort(candidates.begin(), candidates.end());
        backtracing(candidates, target, 0, 0, used);
        return result;
    }
};
```

## （复习）131. 分割回文串（C++回溯实现）

```

```
