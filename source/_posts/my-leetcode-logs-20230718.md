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

## 701. 二叉搜索树中的插入操作（迭代遍历二叉搜索树实现插入节点，C++）

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
        //使用迭代法实现
        if(root == NULL){
            TreeNode* node = new TreeNode(val);
            return node;
        }

        TreeNode* curr = root;
        TreeNode* parent = root;
        while(curr != NULL){//while循环当curr等于NULL时弹出，就是需要插入的节点位置
            parent = curr;
            if(curr->val < val){
                curr = curr->right;
            }else{
                curr = curr->left;
            }
        }

        //处理parent和新插入的节点位置的关系
        TreeNode* node = new TreeNode(val);
        if(parent->val > val){//表示插入点在parent的左
            parent->left = node;
        }else{
            parent->right = node;
        }

        return root;
    }
};
```
