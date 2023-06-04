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

## 225.用队列实现栈
```
class MyStack {

    Queue<Integer> q1;
    Queue<Integer> q2;

    public MyStack() {
        q1 = new LinkedList<>();
        q2 = new LinkedList<>();
    }
    
    public void push(int x) {
        //先放在q2辅助队列中，为了保证最后进入的元素最先出来
        q2.offer(x);
        //将q1队列中的其他元素加入到q2中
        while(!q1.isEmpty()){
            q2.offer(q1.poll());
        }
        //最后将q2和q1进行交换
        Queue<Integer> qTemp = q1;
        q1 = q2;
        q2 = qTemp;
    }

    public int pop() {
        return q1.poll();
    }
    
    public int top() {
        return q1.peek();
    }
    
    public boolean empty() {
        return q1.isEmpty();
    }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
```

## 20.有效的括号
```
class Solution {
    public boolean isValid(String s) {
        //初始化栈
        Stack<Character> stack = new Stack<>();
        int index = 0;
        //定义dict用于表示匹配关系
        Map<Character, Character> dict = new HashMap<>();
        dict.put('(', ')');
        dict.put('{', '}');
        dict.put('[',']');
        
        for (char ch : s.toCharArray()) {
            if (dict.containsKey(ch)) {
                stack.push(ch);
            } else {
                //如果此时栈为空，那么表示此时符号进栈之后不可能再找到与之匹配的符号，直接返回false；
                //或者栈不为空，但是此时即将入栈的符号和栈顶的符号不匹配，也直接返回false即可；
                if (stack.isEmpty() || dict.get(stack.pop()) != ch) {
                    return false;
                }
            }
        }

        return stack.isEmpty();
    }
}
```