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

## 98. 验证二叉搜索树（迭代法实现）

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
    public boolean isValidBST(TreeNode root) {
        //使用迭代法实现
        Stack<TreeNode> stack = new Stack<>();
        if(root != null){
            stack.push(root);
        }
        TreeNode pre = null;

        //循环迭代
        while(!stack.isEmpty()){
            //得到栈顶结点
            TreeNode curr = stack.peek();
            //判断curr是否为null
            //按照右中左的顺序加入到栈中
            if(curr != null){
                stack.pop();//弹出栈顶结点
                if(curr.right != null){//判断当前结点的右结点是否为null，不为null
                    stack.push(curr.right);
                }
                stack.push(curr);
                stack.push(null);
                if(curr.left != null){
                    stack.push(curr.left);
                }
            }else{//弹出栈顶null（占位）结点
                stack.pop();
                //对结点进行操作
                TreeNode tmp = stack.pop();
                if(pre != null && pre.val >= tmp.val){
                    return false;
                }
                pre = tmp;
            }
        }
        return true;
    }
}
```

## 验证二叉搜索树（递归，C++）

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
    long long max_value = LONG_MIN;
    bool isValidBST(TreeNode* root) {
        if(root == NULL){
            return true;
        }

        bool left = isValidBST(root->left);
        if(max_value < root->val){
            max_value = root->val;
        }else{
            return false;
        }
        bool right = isValidBST(root->right);
        return left && right;
    }
};
```
