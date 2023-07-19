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

## 450. 删除二叉搜索树中的节点（递归实现，C++）

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
    TreeNode* deleteNode(TreeNode* root, int key) {

        //删除二叉搜索树的结点
        if(root == NULL){
            return root;
        }

        //第一种情况，key对应的结点为叶子结点
        if(root -> val == key && root->left == NULL && root->right == NULL){
            delete root;
            return NULL;
        }else if(root->val == key && root->left != NULL && root->right == NULL){//第二种情况，删除节点有左子树
            TreeNode* node = root->left;
            delete root;
            return node;
        }else if(root->val == key && root->left == NULL && root->right != NULL){//第三种情况，删除节点有右子树
            TreeNode* node = root->right;
            delete root;
            return node;
        }else if(root->val == key && root->left != NULL && root->right != NULL){//第四种情况，删除节点左右子树都存在
            TreeNode* curr = root->right;//遍历要删除的结点的右子树
            //找到要删除结点的右子树的最左边子树的结点
            while(curr->left != NULL){
                curr = curr->left;
            }

            //将删除节点的左子树移动到上述curr对应的结点的左子树上
            curr->left = root->left;
            //记录要删除的节点
            TreeNode* tmp = root;
            root = root->right;//返回结点的右子树作为根节点
            delete tmp;
            return root;
        }

        //然后用root->left/right来接着对应的递归返回节点
        if(root->val < key){
            root->right = deleteNode(root->right, key);
        }
        
        if(root->val > key){
            root->left = deleteNode(root->left, key);
        }
        return root;
    }
};
```
