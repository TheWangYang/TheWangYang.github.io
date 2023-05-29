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

## 
