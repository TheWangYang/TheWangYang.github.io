---
title: my-leetcode-logs-20230610
date: 2023-06-17
tags:
- LeetCode
- Java
- alibaba
- 二叉树（从左子叶之和开始）
categories:
- LeetCode Logs
---

## 404.左叶子之和（迭代法）

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
    public int sumOfLeftLeaves(TreeNode root) {
        //使用迭代法实现
        int result = 0;
        if(root == null){
            return result;
        }
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while(!st.isEmpty()){
            TreeNode node = st.peek();
            st.pop();
            //判断下一个结点是不是左叶子结点
            if(node.left != null && node.left.left == null && node.left.right == null){
                result += node.left.val;
            }

            //按照右左中的顺序加入到stack中
            if(node.left != null){
                st.push(node.left);
            }

            if(node.right != null){
                st.push(node.right);
            }
        }
        return result;
    }
}
```

##
