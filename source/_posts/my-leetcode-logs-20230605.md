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


## 145.二叉树的后序遍历
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
    //定义后续遍历递归函数
    void treePostOrderTraversal(TreeNode root, List<Integer> result){
        if(root == null){
            return;
        }
        treePostOrderTraversal(root.left, result);
        treePostOrderTraversal(root.right, result);
        result.add(root.val);
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        treePostOrderTraversal(root, result);
        return result;
    }
}
```

## 94.二叉树的中序遍历
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
    void treeInorderTraversal(TreeNode root, List<Integer> result){
        if(root == null){
            return;
        }

        treeInorderTraversal(root.left, result);
        result.add(root.val);
        treeInorderTraversal(root.right, result);
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> result = new ArrayList<>();
        treeInorderTraversal(root, result);
        return result;
    }
}
```

## 144.二叉树的前序遍历（迭代方式）
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
    public List<Integer> preorderTraversal(TreeNode root) {
        //非递归方式遍历二叉树
        //定义一个保存节点值的栈
        Stack<TreeNode> st = new Stack<>();
        //定义返回的结果数组
        List<Integer> result = new ArrayList<>();

        if(root == null){
            return result;
        }
        
        st.add(root);
        //while循环Stack栈，将其中的val添加到result数组中
        while(!st.isEmpty()){
            TreeNode tmpNode = st.pop();
            //将当前节点加入到result中
            result.add(tmpNode.val);
            //然后将tmpNode节点的右节点现加入到st中
            if(tmpNode.right != null){
                st.push(tmpNode.right);
            }

            if(tmpNode.left != null){
                st.push(tmpNode.left);
            }
        }
        return result;
    }
}
```

## 