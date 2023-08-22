---
title: my-leetcode-logs-20230727
date: 2023-07-27 13:28:15
tags:
- LeetCode
- 回溯算法章节
- C++
- Alibaba
categories:
- LeetCode Logs
---

## （复习）108. 将有序数组转换为二叉搜索树

```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        //使用迭代法实现
        if(nums.size() == 0){
            return NULL;
        }

        //创建保存结点的队列
        queue<TreeNode*> nodeQ;
        //创建保存左left边界的队列
        queue<int> leftQ;
        //创建保存右right边界的队列
        queue<int> rightQ;

        //初始化一个空结点并加入到队列中
        TreeNode* root = new TreeNode(0);
        nodeQ.push(root);

        //将left和right加入到对应的队列中
        leftQ.push(0);
        rightQ.push(nums.size() - 1);

        //循环遍历实现对树的构造
        while(!nodeQ.empty()){
            //首先得到nodeQ队列头部的结点进行赋值
            TreeNode* curr = nodeQ.front();
            nodeQ.pop();
            //得到left和right
            int left = leftQ.front();
            leftQ.pop();

            int right = rightQ.front();
            rightQ.pop();

            int mid = left + (right - left) / 2;

            curr->val = nums[mid];

            //处理左子树
            if(left <= mid - 1){
                curr->left = new TreeNode(0);
                leftQ.push(left);
                rightQ.push(mid - 1);
                nodeQ.push(curr->left);
            }

            //处理右子树
            if(right >= mid + 1){
                curr->right = new TreeNode(0);
                leftQ.push(mid + 1);
                rightQ.push(right);
                nodeQ.push(curr->right);
            }
        }

        return root;

    }
};
```

## 538. 把二叉搜索树转换为累加树（C++递归实现）

```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:

    void convertBSTHelper(TreeNode* root, int &sum) {
        if (root == nullptr) {
            return;
        }

        // 遍历右子树
        convertBSTHelper(root->right, sum);

        // 更新当前节点的值为累加和
        sum += root->val;
        root->val = sum;

        // 遍历左子树
        convertBSTHelper(root->left, sum);
    }

    TreeNode* convertBST(TreeNode* root) {
        //使用递归实现
        int sum = 0;
        convertBSTHelper(root, sum);
        return root;
    }

};
```

## 538. 把二叉搜索树转换为累加树（C++迭代法实现）

```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int pre;//pre保存当前累加值

    //使用迭代法实现
    TreeNode* get_new_BST(TreeNode* root){
        if(root == NULL){
            return NULL;
        }
        TreeNode* curr = root;
        //使用栈保存树的结点，实现迭代遍历
        stack<TreeNode*> st;

        //使用循环实现，按照右中左实现反中序遍历
        while(curr != NULL || !st.empty()){
            if(curr != NULL){
                st.push(curr);
                curr = curr->right;//右
            }else{
                //中
                curr = st.top();
                st.pop();
                curr->val += pre;

                pre = curr->val;//更新累加之后的值

                //左
                curr = curr->left;
            }
        }

        return root;
    }

    TreeNode* convertBST(TreeNode* root) {
        pre = 0;
        TreeNode* resultNode = get_new_BST(root);
        return resultNode;
    }
};
```

## 77. 组合（C++回溯法实现）

```
class Solution {
public:
    //设置path保存当前层的满足题意结果
    vector<int> path;
    //设置保存所有path的result结果数组
    vector<vector<int>> result;
    //使用回溯法实现
    void huisu(int n, int k,int startIndex){
        if(path.size() == k){
            result.push_back(path);
        }

        for(int i = startIndex; i <= n;i ++){
            path.push_back(i);
            //递归调用
            huisu(n, k, i + 1);
            //回溯回来之后，弹出最后一个元素
            path.pop_back();
        }
    }

    vector<vector<int>> combine(int n, int k) {
        //首先清空result数组
        result.clear();

        //path清空
        path.clear();

        //调用回溯法
        huisu(n, k, 1);

        return result;
    }
};
```

## 77. 组合（C++回溯法剪枝实现）

```
class Solution {
public:
    //设置path保存当前层的满足题意结果
    vector<int> path;
    //设置保存所有path的result结果数组
    vector<vector<int>> result;
    //使用回溯法实现
    void huisu(int n, int k,int startIndex){
        if(path.size() == k){
            result.push_back(path);
            return;
        }

        for(int i = startIndex; i <= n - (k - path.size()) + 1;i ++){
            path.push_back(i);
            //递归调用
            huisu(n, k, i + 1);
            //回溯回来之后，弹出最后一个元素
            path.pop_back();
        }
    }

    vector<vector<int>> combine(int n, int k) {
        //首先清空result数组
        result.clear();

        //path清空
        path.clear();

        //调用回溯法
        huisu(n, k, 1);

        return result;
    }
};
```
