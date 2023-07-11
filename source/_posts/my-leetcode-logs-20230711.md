---
title: my-leetcode-logs-20230711
date: 2023-07-11 10:29:25
tags:
- LeetCode
- Java
- alibaba
- 二叉树（从700. 二叉搜索树中的搜索开始）
categories:
- LeetCode Logs
---

## 700. 二叉搜索树中的搜索

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
    public TreeNode searchBST(TreeNode root, int val) {
        if(root == null){
            return null;
        }

        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);

        while(!que.isEmpty()){
            TreeNode node = que.poll();
            if(node.val == val){
                return node;
            }

            if(node.left != null){
                que.offer(node.left);
            }

            if(node.right != null){
                que.offer(node.right);
            }
        }
        return null;
    }
}
```


## 