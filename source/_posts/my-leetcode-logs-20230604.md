---
title: my-leetcode-logs-20230604
date: 2023-06-04 16:52:42
tags:
- LeetCode
- Java
- alibaba
- 双指针
categories:
- LeetCode Logs
---

# 双指针相关题目
## 27.移除元素
```
class Solution {
    public int removeElement(int[] nums, int val) {
        int slow = 0;
        int fast = 0;

        while(fast < nums.length){
            if(nums[fast] != val){
                nums[slow++] = nums[fast];
            }
            fast++;
        }
        return slow;
    }
}
```

*双指针法中出现的题目均为前边几个章节中已经出现过的，这里就不再赘述，可以查看本人之前的博客进行学习。*


# 栈与队列
## 232.用栈实现队列
```
class MyQueue {

    //定义两个栈
    Stack<Integer> stackIn;
    Stack<Integer> stackOut;

    //初始化队列
    public MyQueue() {
        stackIn = new Stack<Integer>();
        stackOut = new Stack<Integer>();
    }

    public void push(int x) {
        stackIn.push(x);
    }
    
    public int pop() {
        //调用得到stackOut
        isStackOut();
        return stackOut.pop();
    }
    
    public int peek() {
        isStackOut();
        return stackOut.peek();
    }
    
    public boolean empty() {
        return stackOut.isEmpty() && stackIn.isEmpty();
    }

    //判断stackOut是否为空，如果是空的，直接将stackIn中的元素放到stackOut中
    private void isStackOut(){
        //如果栈不是空的
        if(!stackOut.isEmpty()){
            return;
        }

        while(!stackIn.isEmpty()){
            stackOut.push(stackIn.pop());
        }
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */
```