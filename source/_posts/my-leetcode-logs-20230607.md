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

## 101.对称二叉树（迭代写法，使用队列实现）

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
    public boolean isSymmetric(TreeNode root) {
        if(root == null){
            return true;
        }

        //使用迭代法进行遍历
        //使用队列每次保存左右子树的同外侧结点和同内侧结点
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root.left);
        que.offer(root.right);

        //while循环遍历que进行比较
        while(!que.isEmpty()){
            //得到que中的两个节点判断是否相同
            TreeNode leftNode = que.peek();
            que.poll();
            TreeNode rightNode = que.peek();
            que.poll();

            //进行判断的逻辑
            if(leftNode == null && rightNode == null){
                continue;//表示两个结点都是空的，那么continue
            }

            //判断两个结点是否相同
            // if(leftNode != null && rightNode == null){
            //     return false;
            // }else if(leftNode == null && rightNode != null){
            //     return false;
            // }else if(leftNode.val != rightNode.val){
            //     return false;
            // }

            if((leftNode == null || rightNode == null || (leftNode.val != rightNode.val))){
                return false;
            }

            //然后将leftNode的左子树和rightNode的右子树加入到que中
            que.offer(leftNode.left);
            que.offer(rightNode.right);

            //将leftNode的右子树和rightNode的左子树加入到que中
            que.offer(leftNode.right);
            que.offer(rightNode.left);
        }
        return true;
    }
}
```

## 104.二叉树的最大深度（递归方法）

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
    //递归方法
    public int getDepth(TreeNode root){
        if(root == null){
            return 0;
        }

        int leftDepth = getDepth(root.left);
        int rightDepth = getDepth(root.right);
        int maxDepth = 1 + Math.max(leftDepth, rightDepth);
        return maxDepth;
    }

    public int maxDepth(TreeNode root) {
        if(root == null){
            return 0;
        }
        return getDepth(root);
    }
}
```

## 104.二叉树的最大深度（使用迭代法，队列实现层次遍历）

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
    public int maxDepth(TreeNode root) {
        //迭代法求二叉树深度
        if(root == null){
            return 0;
        }
        int maxResult = 0;
        //设置队列存储结点
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while(!que.isEmpty()){
            //深度加1
            maxResult++;
            int size = que.size();
            for(int i = 0;i < size;i++){
                TreeNode node = que.peek();
                que.poll();
                if(node.left != null){
                    que.offer(node.left);
                }
                if(node.right != null){
                    que.offer(node.right);
                }
            }
        }
        return maxResult;
    }
}
```

## 559.N 叉树的最大深度

```
/*
// Definition for a Node.
class Node {
    public int val;
    public List<Node> children;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, List<Node> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
    public int maxDepth(Node root) {
        //使用迭代法+队列实现层次遍历
        if(root == null){
            return 0;
        }
        int result = 0;
        //定义队列
        Queue<Node> que = new LinkedList<>();
        que.offer(root);
        while(!que.isEmpty()){
            result++;
            int len = que.size();
            for(int i = 0;i < len;i++){
                Node node = que.peek();
                que.poll();

                for(int j = 0;j < node.children.size();j++){
                    if(node.children.get(j) != null){
                        que.offer(node.children.get(j));
                    }
                }
            }
        }
        return result;
    }
}
```

## 111.二叉树的最小深度（使用递归法）

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
    public int getDepth(TreeNode root, int minDepth){
        if(root == null){
            return 0;
        }

        int leftDepth = getDepth(root.left, minDepth);
        int rightDepth = getDepth(root.right, minDepth);

        if(root.left == null || root.right == null){
            minDepth = 1 + Math.max(leftDepth, rightDepth);
        }else{
            minDepth = 1 + Math.min(leftDepth, rightDepth);
        }
        return minDepth;
    }

    public int minDepth(TreeNode root) {
        if(root == null){
            return 0;
        }
        int result = 0;
        return getDepth(root, result);
    }
}
```

## 111.二叉树的最小深度（使用迭代法+队列实现层次遍历）

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
    public int minDepth(TreeNode root) {
        //迭代法得到最小深度，使用队列实现
        if(root == null){
            return 0;
        }
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        int result = 0;
        while(!que.isEmpty()){
            int len = que.size();
            result++;
            for(int i = 0;i < len;i++){
                TreeNode node = que.peek();
                que.poll();
                if(node.left == null && node.right == null){
                    return result;
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

## 222.完全二叉树的节点个数（使用层次遍历实现）

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
    public int countNodes(TreeNode root) {
        if(root == null){
            return 0;
        }
        int result = 0;
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        result++;
        while(!que.isEmpty()){
            int len = que.size();
            for(int i = 0; i < len;i++){
                TreeNode node = que.peek();
                que.poll();
                if(node.left != null){
                    result++;
                    que.offer(node.left);
                }
                if(node.right != null){
                    result++;
                    que.offer(node.right);
                }
            }
        }
        return result;
    }
}
```

## 110.平衡二叉树（递归方法）

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
    //使用递归方法
    public int getHeight(TreeNode root){
        if(root == null){
            return 0;
        }
        
        int leftHeight = getHeight(root.left);
        if(leftHeight == -1){//表示不满足平衡二叉树
        return -1;
        }
        int rightHeight = getHeight(root.right);
        if(rightHeight == -1){
            return -1;
        }
        int result = Math.abs(leftHeight - rightHeight) > 1 ? -1 : 1 + Math.max(leftHeight, rightHeight);
        return result;
    }
    public boolean isBalanced(TreeNode root) {
        return getHeight(root) == -1 ? false: true;
    }
}
```

## 110.平衡二叉树（迭代法遍历得到以当前结点为根结点的最大高度，然后求每个结点的左右子树的高度差）

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
    //定义求以root结点作为根节点的最大高度
    public int getHeight(TreeNode root){
        if(root == null){
            return 0;
        }
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        int result = 0;
        while(!que.isEmpty()){
            int len = que.size();
            result++;
            for(int i = 0;i < len;i++){
                TreeNode node = que.peek();
                que.poll();
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
    public boolean isBalanced(TreeNode root) {
        //遍历的时候求当前左右节点的最大高度，然后求之间的差的绝对值，将该值和1比较
        //外层使用二叉树的统一迭代遍历
        if(root == null){
            return true;
        }
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while(!st.isEmpty()){
            TreeNode node = st.peek();
            if(node != null){
                st.pop();
                //按照后序遍历方法
                st.push(node);//中
                st.push(null);
                //添加进去之前需要判断是否为平衡树
                int leftHeight = getHeight(node.left);
                int rightHeight = getHeight(node.right);
                if(Math.abs(leftHeight - rightHeight) > 1){
                    return false;
                }
                //右子树
                if(node.right != null){
                    st.push(node.right);
                }
                //左子树
                if(node.left != null){
                    st.push(node.left);
                }
            }else{
                //首先弹出标记用的空结点
                st.pop();
                node = st.peek();
                st.pop();
            }
        }
        return true;
    }
}
```

## 110.平衡二叉树（递归得到树的最大高度+遍历当前结点判断该结点的左右子树高度差是否大于1）

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
    public int getHeight(TreeNode root){
        if(root == null){
            return 0;
        }
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        int maxDepth = 1 + Math.max(leftHeight, rightHeight);
        return maxDepth;
    }
    
    public boolean isBalanced(TreeNode root) {
        //遍历的时候求当前左右节点的最大高度，然后求之间的差的绝对值，将该值和1比较
        //外层使用二叉树的统一迭代遍历
        if(root == null){
            return true;
        }
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while(!st.isEmpty()){
            TreeNode node = st.peek();
            if(node != null){
                st.pop();
                //按照后序遍历方法
                st.push(node);//中
                st.push(null);
                //添加进去之前需要判断是否为平衡树
                int leftHeight = getHeight(node.left);
                int rightHeight = getHeight(node.right);
                if(Math.abs(leftHeight - rightHeight) > 1){
                    return false;
                }
                //右子树
                if(node.right != null){
                    st.push(node.right);
                }
                //左子树
                if(node.left != null){
                    st.push(node.left);
                }
            }else{
                //首先弹出标记用的空结点
                st.pop();
                node = st.peek();
                st.pop();
            }
        }
        return true;
    }
}
```

## 257.二叉树的所有路径

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
    //使用递归调用实现
    //参数分别为传入的结点，本次的单条路径，所有路径结果数组
    public void travelTreeAllPath(TreeNode root, List<Integer> path, List<String> result){
        //将中结点加入到path中，这样才算遍历到了叶子结点
        path.add(root.val);
        //递归条件，到叶子节点结束递归
        if(root.left == null && root.right == null){
            //结束递归的时候将path中对应的结果添加到result list中
            String path_str = "";
            for(int i = 0;i < path.size() - 1;i++){
                path_str += String.valueOf(path.get(i));
                path_str += "->";
            }
            path_str += path.get(path.size() - 1);
            //将当前结果加入到result list中
            result.add(path_str);
            return;
        }

        //每次递归需要执行的代码
        //不是空结点才进行递归
        if(root.left != null){
            travelTreeAllPath(root.left, path, result);
            path.remove(path.size() - 1);//回溯
        }

        if(root.right != null){
            travelTreeAllPath(root.right, path, result);
            path.remove(path.size() - 1);//回溯
        }
    }
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        travelTreeAllPath(root, path, result);
        return result;
    }
}


//使用StringBuilder进行字符串的构造，效率提升很大
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
    //使用递归调用实现
    //参数分别为传入的结点，本次的单条路径，所有路径结果数组
    public void travelTreeAllPath(TreeNode root, List<Integer> path, List<String> result){
        //将中结点加入到path中，这样才算遍历到了叶子结点
        path.add(root.val);
        //递归条件，到叶子节点结束递归
        if(root.left == null && root.right == null){
            //结束递归的时候将path中对应的结果添加到result list中
            StringBuilder sb = new StringBuilder();
            for(int i = 0;i < path.size() - 1;i++){
                sb.append(String.valueOf(path.get(i)));
                sb.append("->");
            }
            sb.append(path.get(path.size() - 1));
            //将当前结果加入到result list中
            result.add(sb.toString());
            return;
        }

        //每次递归需要执行的代码
        //不是空结点才进行递归
        if(root.left != null){
            travelTreeAllPath(root.left, path, result);
            path.remove(path.size() - 1);//回溯
        }

        if(root.right != null){
            travelTreeAllPath(root.right, path, result);
            path.remove(path.size() - 1);//回溯
        }
    }
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        travelTreeAllPath(root, path, result);
        return result;
    }
}
```

## 257.二叉树的所有路径（使用迭代方法实现）

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
    public List<String> binaryTreePaths(TreeNode root) {
        //使用前序迭代法
        List<String> result = new ArrayList<>();
        if(root == null){
            return result;
        }
        //保存当前对应的tmp path
        Stack<String> path = new Stack<>();
        //保存前序遍历时的树结点
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        path.push(String.valueOf(root.val));
        while(!st.isEmpty()){
            TreeNode node = st.peek();
            st.pop();//弹出栈顶结点

            //去除该节点对应的path
            String str = path.peek();
            path.pop();

            //入栈之前先判断当前是否为叶子结点
            if(node.left == null && node.right == null){
                //将path放入到result数组中
                result.add(str);
            }

            //右左中顺序入栈
            if(node.right != null){
                st.push(node.right);
                StringBuilder sb = new StringBuilder(str);
                sb.append("->");
                sb.append(String.valueOf(node.right.val));
                path.push(sb.toString());
            }
            
            if(node.left != null){
                st.push(node.left);
                StringBuilder sb = new StringBuilder(str);
                sb.append("->");
                sb.append(String.valueOf(node.left.val));
                path.push(sb.toString());
            }
        }
        return result;
    }
}
```

## 404.左叶子之和（使用迭代法）

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
        //二叉树迭代遍历得到左右左叶子之和
        if(root == null){
            return 0;
        }
        int result = 0;
        Stack<TreeNode> st = new Stack<>();
        st.push(root);
        while(!st.isEmpty()){
            TreeNode node = st.peek();
            if(node != null){
                st.pop();
                //判断条件需要重新理解
                if(node.left != null && node.left.left == null && node.left.right == null){
                    result += node.left.val;
                }
                //按照右左中的顺序加入栈中
                if(node.right != null){
                    st.push(node.right);
                }
                if(node.left != null){
                    st.push(node.left);
                }
                st.push(node);
                st.push(null);
            }else{
                st.pop();
                node = st.peek();
                st.pop();
            }
        }
        return result;
    }
}
```


