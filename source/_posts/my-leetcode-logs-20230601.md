---
title: my-leetcode-logs-20230601
date: 2023-06-01 16:08:41
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---


## 350.两个数组的交集
```
class Solution {
    public int[] intersect(int[] nums1, int[] nums2) {
        int[] record1 = new int[1001];
        int[] record2 = new int[1001];

        int nums1Length = nums1.length;
        int nums2Length = nums2.length;

        for(int i = 0; i < nums1Length;i++){
            record1[nums1[i]]++; 
        }

        for(int i = 0; i< nums2Length;i++){
            record2[nums2[i]]++;
        }

        List<Integer> tmp = new ArrayList<Integer>();
        
        int n = Math.max(record1.length, record2.length);
        
        for(int i = 0;i < n;i++){
            if(record1[i] > 0 && record2[i] > 0){
                int m = Math.min(record1[i], record2[i]);
                for(int j = 0;j < m;j++){
                    tmp.add(i);
                }
            }
        }

        int[] result = new int[tmp.size()];
        for(int i = 0; i < tmp.size();i++){
            result[i] = tmp.get(i);
        }
        return result; 
    }
}
```

## 202.快乐数
```
class Solution {
    //得到数字n的每个位置上的数字平方和
    int getSum(int n){
        int sum = 0;
        while(n != 0){
            sum += (n % 10) * (n % 10);
            n = n / 10;
        }
        return sum;
    }

    public boolean isHappy(int n) {
        Map<Integer, Integer> dict = new HashMap<>();

        while(true){
            int currSum = getSum(n);
            if(currSum == 1){
                return true;
            }

            if(dict.containsKey(currSum) && dict.get(currSum) != 0){
                return false;
            }else{
                dict.put(currSum, 1);
            }
            n = currSum;
        }
    }
}
```

## 454.四数相加II
```
class Solution {
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        Map<Integer,Integer> map = new HashMap<>();

        for(int i = 0;i < nums1.length;i++){
            for(int j = 0; j < nums2.length;j++){
                map.put(nums1[i] + nums2[j], map.getOrDefault(nums1[i] + nums2[j], 0) + 1);
            }
        }

        int count = 0;
        for(int i = 0; i < nums3.length;i++){
            for(int j = 0;j < nums4.length;j++){
                count += map.getOrDefault(0 - (nums3[i] + nums4[j]), 0);
            }
        }
        return count;
    }
}
```

## 15.三数之和
```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        //首先对nums进行排序
        Arrays.sort(nums);
        //判断第一个nums是否为大于0，如果是直接返回空的list
        if(nums[0] > 0) return result;
        int n = nums.length;

        //然后使用双指针循环
        for(int i = 0; i < n;i++){
            if(nums[0] > 0){
                return result;
            }

            //从i=1开始，判断当前元素和前一个元素是否相同，相同直接跳过
            if(i > 0 && nums[i] == nums[i - 1]){
                continue;
            }

            //然后使用双指针
            int left = i + 1;
            int right = n - 1;

            //当left<right时进入循环
            while(left < right){
                //判断三者之和与0的大小关系
                //当三者之和大于0时，right--
                if(nums[i] + nums[left] + nums[right] > 0){
                    right--;
                }else if(nums[i] + nums[left] + nums[right] < 0){
                    left++;
                }else{
                    //三者之和等于0，那么作为一个结果添加到result中
                    List<Integer> tmp = new ArrayList<>();
                    tmp.add(nums[i]);
                    tmp.add(nums[left]);
                    tmp.add(nums[right]);
                    result.add(tmp);//将当前的结果list添加到result中

                    //然后去除和left/right重复的元素
                    while(left < right && nums[right] == nums[right - 1]){
                        right--;
                    }

                    while(left < right && nums[left] == nums[left + 1]){
                        left++;
                    }

                    left++;//左指针向右++
                    right--;//右指针向左--
                    
                }
            }

        }
        return result;
    }
}
```

## 18.四数之和
```
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;

        for(int i = 0; i < nums.length - 1;i ++){
            //进行剪枝处理
            if(nums[i] > target && nums[i] >= 0){
                break;
            }

            //去重
            if(i > 0 && nums[i - 1] == nums[i]){
                continue;
            }

            //进入第二层循环
            for(int j = i + 1; j < nums.length;j++){
                //剪枝
                if(nums[i] + nums[j] > target && nums[i] + nums[j] >= 0){
                    break;
                }

                //对j进行去重
                if(j > i + 1 && nums[j - 1] == nums[j]){
                    continue;
                }

                int left = j + 1;
                int right = n - 1;
                //准备使用双指针
                while(left < right){
                    if(nums[i] + nums[j] + nums[left] + nums[right] > target){
                        right--;
                    }else if(nums[i] + nums[j] + nums[left] + nums[right] < target){
                        left++;
                    }else{
                        List<Integer> tmp = new ArrayList<>();
                        tmp.add(nums[i]);
                        tmp.add(nums[j]);
                        tmp.add(nums[left]);
                        tmp.add(nums[right]);
                        result.add(tmp);

                        while(left < right && nums[right] == nums[right - 1]){
                            right--;
                        }

                        while(left < right && nums[left] == nums[left + 1]){
                            left++;
                        }

                        left++;
                        right--;
                    }
                }
            }
        }
        return result;
    }
}
```

## 344.反转字符串
```
class Solution {
    public void reverseString(char[] s) {
        int left = 0;
        int right = s.length - 1;

        while(left < right){
            char tmp = s[left];
            s[left] = s[right];
            s[right] = tmp;
            left++;
            right--;
        }
    }
}
```

## 541.反转字符串 II
```
class Solution {
    void reverse(char[] s_char, int begin, int end){
        while(begin < end){
            char tmp = s_char[end];
            s_char[end] = s_char[begin];
            s_char[begin] = tmp;
            begin++;
            end--;
        }
    }

    public String reverseStr(String s, int k) {
        //将String字符串转换为char数组
        char[] s_char = s.toCharArray();
        int n = s.length();
        int i = 0;

        for(;i < n;i = i + 2*k){
            int start = i;
            int end = Math.min(n - 1, start + k - 1);
            reverse(s_char, start, end);
        }
        
        return new String(s_char);
    }
}
```

## 剑指 Offer 05. 替换空格
```
class Solution {
    public String replaceSpace(String s) {
        if(s == null){
            return null;
        }

        StringBuilder sb = new StringBuilder();
        for(int i = 0;i < s.length();i++){
            if(s.charAt(i) == ' '){
                sb.append("%20");
            }else{
                sb.append(s.charAt(i));
            }
        }
        return sb.toString();
    }
}
```

## 151.反转字符串中的单词
```
class Solution {

    void reverseString(StringBuilder sb, int begin, int end){
        //然后反转字符串
        while(begin < end){
            char tmp = sb.charAt(begin);
            sb.setCharAt(begin, sb.charAt(end));
            sb.setCharAt(end, tmp);
            begin++;
            end--;
        }
    }

    public String reverseWords(String s) {
        StringBuilder sb = new StringBuilder();
        //先删除字符串中前边或后边的空格
        int left = 0;
        int right = s.length() - 1;

        while(s.charAt(left) == ' '){
            left++;
        }

        while(s.charAt(right) == ' '){
            right--;
        }

        //去除字符串中间多余的空格
        while(left <= right){
            if(s.charAt(left) != ' ' || sb.charAt(sb.length() - 1) != ' '){
                sb.append(s.charAt(left));
            }
            left++;
        }

        int begin = 0;
        int end = sb.length() - 1;
        reverseString(sb, begin, end);

        //反转字符串之后，反转字符串中每个单词
        int start = 0;
        int inner_end = 1;
        int n = sb.length();
        while(start < n){
            while(inner_end < n && sb.charAt(inner_end) != ' '){
                inner_end++;
            }
            //反转从start到end的字符串
            reverseString(sb, start, inner_end - 1);
            //然后进入下一个单词
            start = inner_end + 1;
            inner_end = start + 1;
        }
        return sb.toString();
    }
}
```

## 剑指Offer58-II.左旋转字符串
```
class Solution {
    public String reverseLeftWords(String s, int n) {
        int l = s.length();
        //得到实际左旋转的格数
        int k = n % l;

        StringBuilder sb = new StringBuilder();
        for(int i = k;i < l;i++){
            sb.append(s.charAt(i));
        }

        for(int i = 0;i < k;i++){
            sb.append(s.charAt(i));
        }

        return sb.toString();
    }
}
```

## 28.找出字符串中第一个匹配项的下标
```
class Solution {
    public int strStr(String haystack, String needle) {
        if(haystack.length() < needle.length()) return -1;
        int i = 0;
        int j = 0;
        int result = -1;
        while(i < haystack.length()){
            //每次设置j从0开始
            j = 0;
            result = i;
            //判断是否成功，成功直接返回true，否则返回false
            if(haystack.charAt(i) == needle.charAt(j)){
                while(j < needle.length() && i < haystack.length()){
                    if(haystack.charAt(i) == needle.charAt(j)){
                        i++;
                        j++;
                        continue;
                    }else{
                        break;
                    }
                }
                if(j == needle.length()){
                    return result;
                }
            }
            i = result + 1;
        }
        return -1;
    }
}
```

## 