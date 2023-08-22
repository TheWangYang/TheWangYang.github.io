---
title: my-leetcode-logs-20230726
date: 2023-07-26 12:14:58
tags:
- LeetCode
- C++
- Alibaba
categories:
- LeetCode Logs
---

## 复习：669. 修剪二叉搜索树（迭代法实现，C++）

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
        //修剪二叉树，迭代实现
        if(root == NULL){
            return NULL;
        }

        //修剪root节点
        while(root != NULL && (root->val < low || root->val > high)){
            if(root->val < low){
                root = root -> right;
            }else{
                root = root -> left;
            }
        }


        //修剪root结点的左子树小于low的情况
        TreeNode* curr = root;
        while(curr != NULL){
            //循环找到左边界
            while(curr->left != NULL && curr -> left -> val < low){
                curr -> left = curr -> left -> right;
            }
            curr = curr -> left;
        }

        curr = root;
        while(curr != NULL){
            //循环处理右子树大于high的情况
            while(curr->right != NULL && curr->right->val > high){
                curr->right = curr -> right -> left;
            }
            curr = curr->right;
        }

        return root;
    }
};
```

## 108. 将有序数组转换为二叉搜索树（C++递归法实现）

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

    //使用递归法实现
    TreeNode* digui(vector<int>& nums, int left, int right){
        if(left > right){
            return NULL;
        }

        //单层递归逻辑
        int mid = left + ((right - left) / 2);
        TreeNode* root = new TreeNode(nums[mid]);

        //递归调用
        root->left = digui(nums, left, mid - 1);
        root->right = digui(nums, mid + 1, right);

        return root;
    }

    TreeNode* sortedArrayToBST(vector<int>& nums) {
        //使用迭代法实现
        return digui(nums, 0, nums.size() - 1);    
    }
};
```

## 108. 将有序数组转换为二叉搜索树（C++迭代法实现）

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

        //初始化根结点
        TreeNode* root = new TreeNode(0);
        queue<TreeNode*> nodeQueue;//创建队列保存树结点
        queue<int> leftQueue;//创建保存left左下标的队列
        queue<int> rightQueue;//创建保存right右下标的队列

        //将left=0和right=nums.size()-1分别放到left和right队列中
        nodeQueue.push(root);
        leftQueue.push(0);
        rightQueue.push(nums.size() - 1);

        //使用while循环
        while(!nodeQueue.empty()){
            //得到当前结点
            TreeNode* curr = nodeQueue.front();
            nodeQueue.pop();

            //拿到left和right下标
            int left = leftQueue.front();
            leftQueue.pop();
            int right = rightQueue.front();
            rightQueue.pop();

            int mid = left + ((right - left) / 2);

            curr->val = nums[mid];//将nums[mid]值复制给curr->val


            //处理左子树
            if(left <= mid - 1){
                //向nodeQueue中加入left空结点，用于下次赋值
                curr->left = new TreeNode(0);
                nodeQueue.push(curr->left);
                leftQueue.push(left);
                rightQueue.push(mid - 1);
            }

            //处理右子树
            if(right >= mid + 1){
                //向nodeQueue中加入right空结点，用于下次赋值
                curr->right = new TreeNode(0);
                nodeQueue.push(curr->right);
                rightQueue.push(right);
                leftQueue.push(mid + 1);
            }
            
        }

        return root;

    }
};
```
