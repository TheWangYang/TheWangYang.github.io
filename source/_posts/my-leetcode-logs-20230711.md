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

## 700. 二叉搜索树中的搜索（层序遍历法）

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

## 700. 二叉搜索树中的搜索（递归法）

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
    public TreeNode huisu(TreeNode root, int val){
        if (root == null || root.val == val) {
            return root;
        }

        //表示在root的左子树中
        if(val < root.val){
            return huisu(root.left, val);
        }else{
            return huisu(root.right, val);
        }

    }

    public TreeNode searchBST(TreeNode root, int val) {
        return huisu(root, val);
    }
}
```

## 98. 验证二叉搜索树（递归法实现）

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
    List<Integer> result = new ArrayList<>();
    //递归实现中序遍历
    public void digui(TreeNode root){
        if(root == null){
            return;
        }

        //中序遍历：右中左
        digui(root.left);
        result.add(root.val);
        digui(root.right);
    }

    public boolean isValidBST(TreeNode root) {
        digui(root);
        //使用中序遍历，同时保存树的结点的值，判断是否为升序即可
        for(int i = 1;i < result.size(); i ++){
            if(result.get(i) <= result.get(i - 1)){
                return false;
            }
        }
        return true;
    }
}
```


## 