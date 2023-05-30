---
title: my-leetcode-logs-20230528
date: 2023-05-28 12:06:37
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---


## 203.移除链表元素

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeElements(ListNode head, int val) {
        ListNode newHead = new ListNode();
        ListNode resultHead = newHead;
        newHead.next = head;

        while(head != null){
            if(head.val != val){
                newHead.next = head;
                newHead = newHead.next;
            }
            head = head.next;
        }
        newHead.next = null;
        return resultHead.next;
    }
}
```

## 707.设计链表

```
//定义链表的结点类
class ListNode{
     int val;//定义链表中的值
     ListNode next;
     //创建构造函数
     ListNode(){}
     //创建自定义构造函数
     ListNode(int val){
         this.val = val;
     }
}

class MyLinkedList {
    //定义链表的成员变量
    int size;
    //定义一个虚拟的头结点
    ListNode head;

    //在默认构造函数中初始化链表
    public MyLinkedList() {
        this.size = 0;
        this.head = new ListNode(0);
    }
    
    public int get(int index) {
        //首先判断index是否无效，如果无效返回-1
        if(index < 0 || index >= this.size){
            return -1;
        }
        ListNode resultNode = this.head;
        for(int i = 0; i <= index;i ++){
            resultNode = resultNode.next;
        }
        return resultNode.val; 
    }
    
    public void addAtHead(int val) {
        addAtIndex(0, val);
    }
    
    public void addAtTail(int val) {
        addAtIndex(this.size, val);
    }
    
    public void addAtIndex(int index, int val) {
        if(index > this.size){
            return;
        }

        if(index < 0){
            index = 0;
        }

        this.size++;

        ListNode predNode = this.head;
        //得到predNode（要插入结点的前驱）
        for(int i = 0;i < index;i++){
            predNode = predNode.next;
        }

        ListNode addNode = new ListNode(val);
        addNode.next = predNode.next;
        predNode.next = addNode;
    }
    
    public void deleteAtIndex(int index) {
        if(index >= this.size || index < 0){
            return;
        }
        this.size--;
        //判断index是否为0
        if(index == 0){
            this.head = this.head.next;
            return;
        }

        ListNode predNode = head;
        //找到需要删除结点的前驱结点
        for(int i = 0; i < index;i++){
            predNode = predNode.next;
        }

        predNode.next = predNode.next.next;
    }
}

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.get(index);
 * obj.addAtHead(val);
 * obj.addAtTail(val);
 * obj.addAtIndex(index,val);
 * obj.deleteAtIndex(index);
 */
```

## 206.反转链表

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode currNode = head;
        ListNode resultNode = null;
        ListNode tmpNode = null;

        while(currNode != null){
            tmpNode = currNode.next;
            currNode.next = resultNode;
            resultNode = currNode;
            currNode = tmpNode;
        }
        return resultNode;
    }
}
```

## 24.两两交换链表中的结点

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode xuniHead = new ListNode(0);
        xuniHead.next = head;
        ListNode curr = xuniHead;

        while(curr.next != null && curr.next.next != null){
            ListNode tmp = curr.next;
            ListNode tmp1 = curr.next.next.next;

            curr.next = curr.next.next;
            curr.next.next = tmp;
            curr.next.next.next = tmp1;

            curr = curr.next.next;
        }
        return xuniHead.next;
    }
}
```

## 19. 删除链表的倒数第 N 个结点

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode fast = dummy;
        ListNode slow = dummy;

        //先移动快指针
        for(int i = 0;i <= n;i++){
            fast = fast.next;
        }
        
        //同时移动快慢指针
        while(fast != null){
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;

        return dummy.next;
    }
}
```

## 面试题02.07.链表相交

```
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode currA = headA;
        ListNode currB = headB;
        int lenA = 0;
        int lenB = 0;

        while(currA != null){
            currA = currA.next;
            lenA++;
        }

        while(currB != null){
            currB = currB.next;
            lenB++;
        }

        currA = headA;
        currB = headB;

        //使得currA指向较长链表的那个
        if(lenA < lenB){
            int tmpL = lenA;
            lenA = lenB;
            lenB = tmpL;
            
            ListNode tmpNode = currA;
            currA = currB;
            currB = tmpNode;
        }

        //求两者的长度差
        int gap = lenA - lenB;

        //然后使得较长的指针移动到较短的链表的尾部位置
        while(gap-- > 0){
            currA = currA.next;
        }

        //同时移动两个链表的指针
        while(currA != null){
            if(currA == currB){
                return currA;
            }

            currA = currA.next;
            currB = currB.next;
        }

        return null;
    }
}
```


## 实现了
