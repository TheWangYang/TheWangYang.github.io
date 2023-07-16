---
title: my-leetcode-logs-20230711
date: 2023-07-11 10:29:25
tags:
- LeetCode
- C++
- Alibaba
categories:
- LeetCode Logs
---

## 验证二叉搜索树（迭代，C++）

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
    bool isValidBST(TreeNode* root) {
        //使用栈来进行迭代法实现（中序遍历迭代）
        stack<TreeNode*> st;
        TreeNode* currNode = root;
        TreeNode* preNode = NULL;

        //使用同一迭代法遍历树结点
        while(currNode != NULL || !st.empty()){
            if(currNode != NULL){
                //将当前结点加入到栈中
                st.push(currNode);
                currNode = currNode -> left;//左
            }else{
                currNode = st.top();
                st.pop();
                //中
                if(preNode != NULL && currNode->val <= preNode->val){
                    return false;
                }
                //保存前一个结点
                preNode = currNode;
                //右
                currNode = currNode -> right;
            }
        }
        return true;
    }
};
```
