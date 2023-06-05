---
title: my-leetcode-logs-20230605
date: 2023-06-05 13:41:52
tags:
- LeetCode
- Java
- alibaba
- 二叉树
categories:
- LeetCode Logs
---

## 144.二叉树的前序遍历
```
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    //定义递归函数
    void preorderTraversal(TreeNode root, List<Integer> result){
        if(root == null){
            return;
        }

        //将当前root节点的值存入到result中
        result.add(root.val);
        preorderTraversal(root.left, result);
        preorderTraversal(root.right, result);
    }
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        preorderTraversal(root, result);
        return result;
    }
}
```
