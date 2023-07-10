---
title: my-leetcode-logs-20230710.md
date: 2023-07-10 11:43:31
tags:
- LeetCode
- Java
- alibaba
- 二叉树（从左子叶之和开始）
categories:
- LeetCode Logs
---

## 513.找树左下角的值（递归写法）

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
    int max_depth = -1;
    int result = 0;

    public void huisu(TreeNode root, int depth){
        //设置递归终止条件
        if(root.left == null && root.right == null){
            if(max_depth < depth){
                max_depth = depth;
                result = root.val;
            }
            return;
        }

        //递归左子树条件
        if(root.left != null){
            depth++;
            huisu(root.left, depth);
            depth--;
        }

        //递归右子树
        if(root.right != null){
            depth++;
            huisu(root.right, depth);
            depth--;
        }
    }

    public int findBottomLeftValue(TreeNode root) {
        huisu(root, 0);
        return result;
    }
}

```

## 513.找树左下角的值（迭代法）

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
    public int findBottomLeftValue(TreeNode root) {
        int result = 0;
        //使用二叉树层序遍历
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        ///如果队列不是空的，那么进入循环
        while(!que.isEmpty()){
            //得到当前队列的长度
            int size = que.size();
            for(int i = 0;i < size;i ++){
                //得到每层的结点
                TreeNode node = que.poll();
                if(i == 0){
                    result = node.val;
                }
                if(node.left != null){
                    que.offer(node.left);
                }
                if(node.right != null){
                    que.offer(node.right);
                }
            }
        }
        return result;
    }
}
```

## 112.路经总和（递归法）

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
    //设置的回溯算法
    public boolean huisu(TreeNode root, int currSum){
        //判断是否为targetSum
        //遇到叶子结点且currSum==0，这表示找到了满足题意的结果
        if(root.left == null && root.right == null && currSum == 0){
            return true;
        }

        //遇到叶子节点直接返回false
        if(root.left == null && root.right == null){
            return false;
        }

        if(root.left != null){
            currSum -= root.left.val;
            if(huisu(root.left, currSum)) return true;
            currSum += root.left.val;
        }

        if(root.right != null){
            currSum -= root.right.val;
            if(huisu(root.right, currSum)) return true;
            currSum += root.right.val;
        }

        return false;
    }
    
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null){
            return false;
        }
        return huisu(root, targetSum - root.val);
    }
}
```

## 
