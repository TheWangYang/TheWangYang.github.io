---
title: my-leetcode-logs
date: 2023-05-24 15:03:57
updated: 2023-05-25
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---


# My LeetCode HOT 100 logs

*use language: java*

## 1.两数之和

```
class Solution {
    public int[] twoSum(int[] nums, int target) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (nums[i] + nums[j] == target) {
                    return new int[]{i, j};
                }
            }
        }
        return new int[0];
    }
}
```

## 49.字母异位词分组

```
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for(String str: strs){
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```

## 128.最长连续序列

```
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for(int num : nums){
            set.add(num);
        }

        int result = 0;

        for(int num : nums){
            if(!set.contains(num - 1)){
                int currNum = num;
                int inner_result = 1;

                while(set.contains(currNum + 1)){
                    inner_result += 1;
                    currNum += 1;
                }
                
                result = Math.max(result, inner_result);
            }
        }
        return result;
    }
}

```

## 283.移动零

```
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        
        int lp = 0;
        int rp = lp;

        while(lp != n - 1){
            if(nums[lp] == 0){
                while(rp != n){
                    if(nums[rp] != 0){
                        int tmp = nums[rp];
                        nums[rp] = nums[lp];
                        nums[lp] = tmp;
                        break;
                    }
                    rp += 1;
                }
            }
            lp += 1;
            rp = lp;
        }
    }
}
```

## 11.盛最多水的容器

```
class Solution {
    public int maxArea(int[] height) {
        int n = height.length;
        int left = 0;
        int right = n - 1;

        int area = 0;
        while(left < right){
            int h = Math.min(height[left], height[right]);
            area = Math.max(area, h * (right - left));
            if(height[left] < height[right]){
                left ++;
            }else{
                right -- ;
            }
        }
        return area;
    }
}
```

## 15.三数之和

```
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        //首先先排序（升序）
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;

        for(int i = 0; i < n; i++){
            if(nums[i] > 0){
                return result;
            }

            //判断i位置的元素是否和其前一个元素相同，相同那么进入下一次循环
            if(i > 0 && nums[i] == nums[i - 1]){
                continue;
            }

            int left = i + 1;
            int right = n - 1;

            while(left < right){
                if(nums[i] + nums[left] + nums[right] > 0){
                    right--;
                }else if(nums[i] + nums[left] + nums[right] < 0){
                    left++;
                }else{
                    List<Integer> tmp = new ArrayList<>();
                    tmp.add(nums[i]);
                    tmp.add(nums[left]);
                    tmp.add(nums[right]);
                    result.add(tmp);

                    while(right > left && nums[right - 1] == nums[right]){
                        right--;
                    }

                    while(right > left && nums[left + 1] == nums[left]){
                        left++;
                    }

                    right--;
                    left++;

                }
            }
        }
        return result;
    }
}
```

## 3.无重复字符的最长子串

```
class Solution {
    public int lengthOfLongestSubstring(String s) {
        int n = s.length();
        if(n == 0) return 0;
        if(n == 1) return 1;

        int left = 0;
        int right = 1;

        int result = 0;
        while(left < n - 1){
            Map<Character, Integer> map = new HashMap<>();
            map.put(s.charAt(left), left);
            while(right < n && !map.containsKey(s.charAt(right))){
                map.put(s.charAt(right), right);
                right++;
            }
            result = Math.max(result, right - left);
            left = left + 1;
            right = left + 1;
        }
        return result;
    }
}
```

## 438. 找到字符串中所有字母异位词

```
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        int n = p.length();
        char[] p_chars = p.toCharArray(); // 转换为字符数组
        Arrays.sort(p_chars); // 对字符数组进行排序
        String sorted_p = new String(p_chars); // 将字符数组转换回字符串
        //System.out.println("sorted_p: " + sorted_p);

        List<Integer> result = new ArrayList<>();
        int left = 0;
        int l = s.length();

        while(left <= l - n){
            String tmp = s.substring(left, left + n);
            char[] tmp_chars = tmp.toCharArray(); // 转换为字符数组
            Arrays.sort(tmp_chars); // 对字符数组进行排序
            String sorted_tmp = new String(tmp_chars); // 将字符数组转换回字符串
            //System.out.println("sorted_tmp: " + sorted_tmp);
            if(sorted_tmp.equals(sorted_p)){
                result.add(left);
            }
            left += 1;
        }
        return result;
    }
}
```

## 560.和为 K 的子数组

```
class Solution {
    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int sum = 0;
        int result = 0;
        map.put(0, 1);
        for(int i = 0;i < nums.length;i++){
            sum += nums[i];

            if(!map.containsKey(sum - k)){
                map.put(sum - k, 0);
            }

            result += map.get(sum - k);

            
            if(!map.containsKey(sum)){
                map.put(sum, 0);
            }

            map.put(sum, map.get(sum) + 1);
        }
        return result;
    }
}
```

## 704.二分查找

```
class Solution {
    public int search(int[] nums, int target) {
        if(target < nums[0] || target > nums[nums.length - 1]){
            return -1;
        }
        
        int left = 0;
        int right = nums.length - 1;
        while(left <= right){
            int mid = (left + right) / 2;
            if(nums[mid] < target){
                left = mid + 1;
            }else if(nums[mid] > target){
                right = mid - 1;
            }else{
                return mid;
            }
        }
        return -1;
    }
}
```

## 35.搜索插入位置

```
class Solution {
    public int searchInsert(int[] nums, int target) {
        if(target < nums[0]){
            return 0;
        }

        if(target > nums[nums.length - 1]){
            return nums.length;
        }

        int left = 0;
        int right = nums.length - 1;
        int mid = -1;

        while(left <= right){
            mid = (left + right) / 2;
            if(nums[mid] == target){
                return mid;
            }else if(nums[mid] < target){
                left = mid + 1;
            }else if(nums[mid] > target){
                right = mid - 1;
            }
        }
        return right + 1;
    }
}
```

## 27.移除元素

```
//暴力解法
class Solution {
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        for(int i = 0; i < n; i ++){
            if(nums[i] == val){
                for(int j = i + 1; j < n; j++){
                    nums[j - 1] = nums[j];
                }
                i--;
                n--;
            }
        }
        return n;
    }
}


//快慢指针法
class Solution {
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int slow = 0;
        for(int fast = 0; fast < n; fast ++){
            if(nums[fast] != val){
                nums[slow++] = nums[fast];
            } 
        }
        return slow;
    }
}

```

## 26.删除有序数组中的重复项

```
//快慢指针
class Solution {
    public int removeDuplicates(int[] nums) {
        Map<Integer, Integer> dict = new HashMap<>();
        int slow = 0;
        for(int fast = 0; fast < nums.length;fast++){
            if(!dict.containsKey(nums[fast])){
                nums[slow++] = nums[fast];
                dict.put(nums[fast], 0);
            }
        }
        return slow;
    }
}
```

## 283.移动零

```
//使用快慢指针
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        if(n == 1){
            return;
        }

        int slow = 0;
        for(int fast = 0; fast < nums.length; fast++){
            if(nums[fast] != 0){
                nums[slow ++] = nums[fast];
            }
        }

        for(; slow < n; slow ++){
            nums[slow] = 0;
        }
    }
}
```

## 844.比较含退格的字符串

```
//使用快慢指针
class Solution {
    public boolean backspaceCompare(String s, String t) {
        char [] s_arr = s.toCharArray();
        char [] t_arr = t.toCharArray();

        if(rebuildString(s_arr).equals(rebuildString(t_arr))){
            return true;
        }else{
            return false;
        }
    }

    //重建字符串函数
    String rebuildString(char[] c){
        int slow = 0;
        for(int fast = 0; fast < c.length; fast++){
            if(c[fast] != '#'){
                c[slow++] = c[fast];
            }else{
                if(slow > 0){
                    slow--;
                }
            }
        }
        return new String(c).substring(0,slow);
    }
}
```

## 977.有序数组的平方

```
//使用前后双指针
class Solution {
    public int[] sortedSquares(int[] nums) {
        int start = 0;
       int end = nums.length;
       int[] new_nums = new int[end];
       int i = end - 1;
       end --;
        while(i >= 0){
            new_nums[i--] = nums[start] * nums[start] > nums[end] * nums[end] ? nums[start]*nums[start++] : nums[end]*nums[end--];
        }
        return new_nums;
    }
}
```

##
