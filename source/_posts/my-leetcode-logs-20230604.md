---
title: my-leetcode-logs-20230604
date: 2023-06-04 16:52:42
tags:
- LeetCode
- Java
- alibaba
- 双指针
- 栈与队列
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

## 1047.删除字符串中的所有相邻重复项
```
class Solution {
    public String removeDuplicates(String s) {
        //初始化定义栈
        Stack<Character> stack = new Stack<Character>();

        for(Character ch : s.toCharArray()){
            //判断栈顶元素是否和当前元素相同，相同同时都删除
            if(!stack.isEmpty() && stack.peek() == ch){
                stack.pop();
            }else{
                stack.push(ch);
            }
        }

        //最终得到stack中的字符串
        String result = "";
        while(!stack.isEmpty()){
            result = stack.pop() + result;
        }
        return result; 
    }
}


//优化版本，使用了StringBuilder加快了代码执行的效率
class Solution {
    public String removeDuplicates(String s) {
        Stack<Character> stack = new Stack<>();
        StringBuilder sb = new StringBuilder();

        for (char ch : s.toCharArray()) {
            if (!stack.isEmpty() && stack.peek() == ch) {
                stack.pop();
            } else {
                stack.push(ch);
            }
        }

        while (!stack.isEmpty()) {
            sb.append(stack.pop());
        }

        return sb.reverse().toString();
    }
}

```

## 150.逆波兰表达式求值
```
class Solution {
    public int evalRPN(String[] tokens) {
        //定义保存符号的Stack
        Stack<String> stack = new Stack<>();
        Map<String, Integer> dict = new HashMap<>();
        dict.put("+", 1);
        dict.put("-", 2);
        dict.put("*", 3);
        dict.put("/", 4);

        for(String str : tokens){
            if(dict.containsKey(str)){
                int second = Integer.parseInt(stack.pop());
                int first = Integer.parseInt(stack.pop());
                int result = 0;
                if(dict.get(str) == 1){
                    result = first + second;
                }else if(dict.get(str) == 2){
                    result = first - second;
                }else if(dict.get(str) == 3){
                    result = first * second;
                }else{
                    result = first / second;
                }
                stack.push(String.valueOf(result));
            }else{
                stack.push(str);
            }
        }
        return Integer.parseInt(stack.pop());
    }
}

//优化之后的代码
class Solution {
    public int evalRPN(String[] tokens) {
        Deque<Integer> stack = new LinkedList();
        for (String s : tokens) {
            if ("+".equals(s)) {        // leetcode 内置jdk的问题，不能使用==判断字符串是否相等
                stack.push(stack.pop() + stack.pop());      // 注意 - 和/ 需要特殊处理
            } else if ("-".equals(s)) {
                stack.push(-stack.pop() + stack.pop());
            } else if ("*".equals(s)) {
                stack.push(stack.pop() * stack.pop());
            } else if ("/".equals(s)) {
                int temp1 = stack.pop();
                int temp2 = stack.pop();
                stack.push(temp2 / temp1);
            } else {
                stack.push(Integer.valueOf(s));
            }
        }
        return stack.pop();
    }
}
```


## 239.滑动窗口最大值
```
//定义自己实现的一个基于双端队列的单调队列类
class MyQueue{
    Deque<Integer> dequeue = new LinkedList<>();
    //设置poll方法
    void poll(int val){
        //移除的时候判断当前移除的元素是否和队列的头部相同，相同则弹出
        if(!dequeue.isEmpty() && dequeue.peek() == val){
            dequeue.poll();
        }
    }
    
    //设置的add方法
    //add的时候需要判断和当前队列中的元素的大小关系，需要维持递减的顺序
    void add(int val){
        while(!dequeue.isEmpty() && dequeue.getLast() < val){
            dequeue.removeLast();//移除最后的元素
        }
        //增加到队列中
        dequeue.add(val);
    }

    //获得队列头部元素值
    int peek(){
        return dequeue.peek();
    }
}


class Solution {
    public int[] maxSlidingWindow(int[] nums, int k) {
        if(nums.length == 1){
            return nums;
        }

        int len = nums.length -k + 1;//定义最后结果数组的长度
        int[] result = new int[len];
        int index = 0;//设置的结果数组对应的index索引值

        MyQueue myqueue = new MyQueue();

        //先将前k个元素放入队列中
        for(int i = 0;i < k;i++){
            myqueue.add(nums[i]);
        }

        //得到第一个前k个元素中最大值
        result[index++] = myqueue.peek();

        //循环遍历后边的长度
        for(int i = k;i < nums.length;i++){
            //滑动窗口往后移动一格，首先判断队列中的第一个元素是否需要弹出
            myqueue.poll(nums[i - k]);
            //然后判断，增加的元素是否需要到达队顶
            myqueue.add(nums[i]);
            //记录最大值
            result[index++] = myqueue.peek();
        }
        return result;
    }
}
```