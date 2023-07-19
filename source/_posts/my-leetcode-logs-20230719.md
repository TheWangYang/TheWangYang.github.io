---
title: my-leetcode-logs-20230719
date: 2023-07-19 14:47:40
tags:
- LeetCode
- C++
- Alibaba
categories:
- LeetCode Logs
---

## 669. 修剪二叉搜索树（递归实现，C++）

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
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        //迭代实现
        if(root == NULL){
            return NULL;
        }

        //当前结点的值小于low，那么遍历root的right右子树
        if(root->val < low){
            TreeNode* node = trimBST(root->right, low, high);
            return node;
        }

        //当前结点的值大于high，那么遍历root的左子树
        if(root->val > high){
            TreeNode* node = trimBST(root->left, low, high);
            return node;
        }

        root->left = trimBST(root->left, low, high);
        root->right = trimBST(root->right, low, high);
        return root;
    }
};
```
