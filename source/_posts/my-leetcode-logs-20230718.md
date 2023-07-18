---
title: my-leetcode-logs-20230718
date: 2023-07-18 10:29:25
tags:
- LeetCode
- C++
- Alibaba
categories:
- LeetCode Logs
---

## 701. 二叉搜索树中的插入操作（递归实现，C++）

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
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        //用递归方法实现
        if(root == NULL){
            TreeNode* node = new TreeNode(val);
            return node;
        }

        if(root->val > val){
            root->left = insertIntoBST(root->left, val);
        }

        if(root->val < val){
            root->right = insertIntoBST(root->right,val);
        }

        return root;
    }
};
```
