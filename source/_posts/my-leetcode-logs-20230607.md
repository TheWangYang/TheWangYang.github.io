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

## 226.翻转二叉树（递归法）
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
    public TreeNode invertTree(TreeNode root) {
        //使用递归
        if(root == null){
            return root;
        }
        //交换root的左右结点
        TreeNode tmpNode = root.left;
        root.left = root.right;
        root.right = tmpNode;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
}
```

## 226.翻转二叉树（前序遍历迭代法）
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
    public TreeNode invertTree(TreeNode root) {
        //使用迭代法
        if(root == null){
            return root;
        }

        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        
        while(!st.isEmpty()){
            //弹出栈顶结点
            TreeNode node = st.peek();
            st.pop();
            //交换左右结点
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
            if(node.left != null){
                st.push(node.left);
            }
            if(node.right != null){
                st.push(node.right);
            }
        }
        return root;
    }
}
```

## 226.翻转二叉树（统一迭代法，前序遍历实现）
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
    public TreeNode invertTree(TreeNode root) {
        //使用统一迭代法
        if(root == null){
            return root;
        }
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while(!st.isEmpty()){
            TreeNode curr = st.peek();
            if(curr != null){
                st.pop();//弹出结点
                //按照右中左进栈（前序遍历）
                if(curr.right != null){
                    st.push(curr.right);
                }
                //中结点入栈
                st.push(curr);
                st.push(null);
                if(curr.left != null){
                    st.push(curr.left);
                }
            }else{
                st.pop();//先弹出null结点
                curr = st.peek();
                st.pop();
                //交换结点
                TreeNode tmp = curr.left;
                curr.left = curr.right;
                curr.right = tmp;
            }
        }
        return root;
    }
}
```

## 101.对称二叉树（递归写法）
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
    //设置递归函数，传入参数分别为root的左右结点
    private boolean compare(TreeNode left, TreeNode right){
        //确定终止条件
        if(left != null && right == null){
            return false;
        }else if(left == null && right != null){
            return false;
        }else if(left == null && right == null){
            return true;
        }else if(left.val != right.val){
            return false;
        }

        //确定递归的内容
        //传入为左节点的左子树和右节点的右子树
        boolean outside = compare(left.left, right.right);
        //传入为左节点的右子树和右节点的左子树
        boolean inside = compare(left.right, right.left);
        boolean eq = outside && inside;
        return eq;
    } 

    public boolean isSymmetric(TreeNode root) {
        //使用递归实现
        if(root == null){
            return true;
        }
        return compare(root.left, root.right);
    }
}
```