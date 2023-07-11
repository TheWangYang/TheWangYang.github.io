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

## 112.路径总和（迭代法）

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

 //自定义类
public class MyNode{
     TreeNode treeNode;
     int currSum = 0;
     MyNode(TreeNode root, int currSum){
         this.treeNode = root;
         this.currSum = currSum;
     }
}

class Solution {
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if(root == null){
            return false;
        }
        //使用非递归方法做
        Stack<MyNode> st = new Stack<>();
        MyNode rootNode = new MyNode(root, root.val);
        st.push(rootNode);

        while(!st.isEmpty()){
            MyNode node = st.peek();
            st.pop();

            if(targetSum == node.currSum && node.treeNode.left == null && node.treeNode.right == null){
                return true;
            }

            //压入右结点栈中
            if(node.treeNode.right != null){
                st.push(new MyNode(node.treeNode.right, node.currSum + node.treeNode.right.val));
            }

            //将左结点压入栈中
            if(node.treeNode.left != null){
                st.push(new MyNode(node.treeNode.left, node.currSum + node.treeNode.left.val));
            }
        }
        return false;
    }
}
```

## 113.路经总和2（回溯法）

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
    //使用回溯方法做
    public void huisu(TreeNode root, int targetSum, List<List<Integer>> result, List<Integer> path){
        if(root == null){
            return;
        }
        path.add(root.val);
        if(root.left == null && root.right == null && targetSum == root.val){
            result.add(new ArrayList(path));
        }

        huisu(root.left, targetSum - root.val, result, path);
        huisu(root.right, targetSum - root.val, result, path);

        path.remove(path.size() - 1);
    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        huisu(root, targetSum, result, path);
        return result;
    }
}
```

## 113.路经总和2（DFS法）

```
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<Integer> cur = new ArrayList<>();
        dfs(root, cur, 0, sum);            
        return res;
    }   

    public void dfs(TreeNode node, List<Integer> cur, int sum, int target){
        if(node == null){
            return ;
        }
        if(node.left == null && node.right == null && node.val + sum == target){
            cur.add(node.val);
            res.add(new ArrayList<>(cur));
            cur.remove(cur.size() - 1);
            return ;
        }            
        cur.add(node.val);
        dfs(node.left, cur, sum + node.val, target);
        dfs(node.right, cur, sum + node.val, target);
        cur.remove(cur.size() - 1);
    }
}
```

## 106. 从中序与后序遍历序列构造二叉树

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
    //参数：中序遍历数组和后续遍历数组
    public TreeNode buildTree(int[] inorder, int[] postorder) {
        //获得分割结点
        int in_length = inorder.length;
        int post_length = postorder.length;

        //判断是否为空结点
        if(in_length == 0 || post_length == 0){
            return null;
        }

        //通过后续序列找到切割结点
        int root_val = postorder[post_length - 1];
        //构造根结点
        TreeNode root = new TreeNode(root_val);
        int k = 0;
        //遍历中序序列，找到切割结点在其中的位置
        for(int i = 0; i < in_length;i++){
            if(root_val == inorder[i]){
                k = i;
                break;
            }
        }

        //按照分割结点将中序序列和后续序列进行分割
        int[] left_in = Arrays.copyOfRange(inorder, 0, k);
        int[] left_post = Arrays.copyOfRange(postorder, 0, k);
        //递归调用函数
        root.left = buildTree(left_in, left_post);

        //按照分割结点构造右子树
        int[] right_in = Arrays.copyOfRange(inorder, k + 1, in_length);
        int[] right_post = Arrays.copyOfRange(postorder, k, post_length - 1);
        root.right = buildTree(right_in, right_post);

        return root;
    }
}
```

## 105. 从前序与中序遍历序列构造二叉树

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
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int pre_length = preorder.length;
        int in_length = inorder.length;
        if(pre_length == 0 || in_length == 0){
            return null;
        }

        int root_val = preorder[0];
        TreeNode root = new TreeNode(root_val);
        
        //得到左子树对应的前序和中序序列
        int k = 0;
        for(int i = 0;i < in_length;i++){
            if(root_val == inorder[i]){
                k = i;
                break;
            }
        }

        //构造左子树
        int[] left_pre = Arrays.copyOfRange(preorder, 1, k + 1);
        int[] left_in = Arrays.copyOfRange(inorder, 0, k);
        root.left = buildTree(left_pre, left_in);

        //构造右子树
        int[] right_pre = Arrays.copyOfRange(preorder, k + 1, pre_length);
        int[] right_in = Arrays.copyOfRange(inorder, k + 1, in_length);
        root.right = buildTree(right_pre, right_in);

        return root;
    }
}
```

## 654. 最大二叉树

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
    public TreeNode constructMaximumBinaryTree(int[] nums) {
        int n_length = nums.length;
        if(n_length == 0){
            return null;
        }

        int max_value = -1;
        int k = 0;
        //首先找到最大值和最大值对应的index
        for(int i = 0;i < nums.length;i++){
            if(max_value < nums[i]){
                max_value = nums[i];
                k = i;
            }
        }

        //创建根结点
        TreeNode root = new TreeNode(max_value);
        
        //从根结点左边构造左子树
        int[] left_nums = Arrays.copyOfRange(nums, 0, k);
        int[] right_nums = Arrays.copyOfRange(nums, k + 1, n_length);
        root.left = constructMaximumBinaryTree(left_nums);
        root.right = constructMaximumBinaryTree(right_nums);

        return root;

    }
}
```
