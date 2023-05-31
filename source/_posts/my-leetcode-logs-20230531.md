---
title: my-leetcode-logs-20230531
date: 2023-05-31 11:59:02
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---


## 142.环形链表II
```
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode detectCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;

        while(fast != null && fast.next != null){
            //慢指针每次走一步
            slow = slow.next;
            //快指针每次走两步
            fast = fast.next.next;
            //如果两个指针相遇，那么一个指针从相遇结点出发，一个指针从头节点出发，等到下次两个节点相遇的时候就是链表环形入口结点
            if(slow == fast){
                ListNode index1 = fast;
                ListNode index2 = head;
                while(index1 != index2){
                    index1 = index1.next;
                    index2 = index2.next;
                }
                return index2;
            }
        }
        return null;
    }
}
```

## 242.有效的字母异位词
```
class Solution {
    public boolean isAnagram(String s, String t) {
        //设置record数组的长度大小为26（26个小写字母）
        int[] record = new int[26];

        for(int i = 0; i< s.length();i++){
            record[s.charAt(i) - 'a'] ++;
        }

        for(int i = 0;i < t.length();i++){
            record[t.charAt(i) - 'a']--;
        }

        for(int i = 0; i < record.length;i++){
            if(record[i] != 0){
                return false;
            }
        }

        return true;
    }
}
```

## 383.赎金信
```
class Solution {
    public boolean canConstruct(String ransomNote, String magazine) {
        int[] record = new int[26];
        
        for(int i = 0; i < ransomNote.length();i++){
            record[ransomNote.charAt(i) - 'a'] ++;
        }

        for(int i = 0;i < magazine.length();i++){
            record[magazine.charAt(i) - 'a']--;
        }

        for(int i = 0; i < record.length;i++){
            if(record[i] > 0){
                return false;
            }
        }
        return true;
    }
}
```

## 349.两个数组的交集
```
class Solution {
    public int[] intersection(int[] nums1, int[] nums2) {
        //数组作为map
        int[] record = new int[1001];
        for(int i = 0; i < nums1.length;i++){
            //map只记录是否有，并不需要记录每个key值对应的values数量
            record[nums1[i]] = 2;
        }

        for(int i = 0; i < nums2.length;i++){
            if(record[nums2[i]] == 2){
                record[nums2[i]] = 3;
            }
        }

        List<Integer> tmpList = new ArrayList<>(); 
        for(int i = 0;i < record.length;i++){
            if(record[i] == 3){
                tmpList.add(i);
            }
        }

        int[] result = new int[tmpList.size()];
        int index = 0;
        for(int num: tmpList){
            result[index] = tmpList.get(index);
            index++;
        }
        return result;
    }
}
```

## 