---
title: my-leetcode-logs-20230607
date: 2023-06-07 11:57:24
tags:
- LeetCode
- Java
- alibaba
- 二叉树
categories:
- LeetCode Logs
---

## 102.二叉树的层序遍历
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
    public List<List<Integer>> levelOrder(TreeNode root) {
        //二叉树层序遍历
        List<List<Integer>> result = new ArrayList<>();

        //BFS搜索
        if(root == null){
            return result;
        }

        //借助队列实现
        Queue<TreeNode> queue = new LinkedList<>();
        //将第一个节点加入队列中
        queue.offer(root);

        //循环进行遍历
        while(!queue.isEmpty()){
            //设置内部的保存结点的list数组
            List<Integer> tmp = new ArrayList<>();
            int size = queue.size();//获得tmp list的长度

            //遍历当前所有的节点
            for(int i = 0;i < size;i++){
                //弹出队列首结点
                TreeNode node = queue.poll();
                //向tmp list中添加弹出结点的val
                tmp.add(node.val);
                //将左右结点加入到queue中
                if(node.left != null){
                    queue.offer(node.left);
                }
                if(node.right != null){
                    queue.offer(node.right);
                }
            }
            //将本次得到的结点list加入到最后的结果list中
            result.add(tmp);
        }
        return result;
    }
}
```
