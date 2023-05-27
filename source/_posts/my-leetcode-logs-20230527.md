---
title: my-leetcode-logs-20230527
date: 2023-05-27 15:28:54
tags:
- LeetCode
- Java
- alibaba
categories:
- LeetCode Logs
---


## 59.螺旋矩阵II

```
class Solution {
    public int[][] generateMatrix(int n) {
        //定义一个动态增加的list，最后转换为int即可
        int[][] matrixResult = new int[n][n];

        //设置给每个格子赋值的值
        int count = 1;//初始值设置为1
        int offset = 1;//设置的每圈应该在右开的时候减少的偏移量
        int startX = 0;
        int startY = 0;
        int loop = n / 2;
        int mid = n / 2;
        int i = 0;
        int j = 0;

        while(loop > 0){
            i = startX;
            j = startY;

            //上行：从左到右进行填充
            for(j = startY;j < n - offset;j++){
                matrixResult[startX][j] = count++;
            }

            //右列：从上到下及逆行填充
            for(i = startX; i < n - offset; i++){
                matrixResult[i][j] = count++;
            }

            for(; j > startY;j--){
                matrixResult[i][j] = count++;
            }

            for(; i > startX;i--){
                matrixResult[i][j] = count++;
            }

            //将对应的startX和startY进行更新
            startX++;
            startY++;

            offset++;
            loop--;
        }

        //最后判断是否需要填充中心位置的元素
        //也就是n为奇数时需要填充
        if(n % 2 != 0){
            matrixResult[mid][mid] = count;
        }
        return matrixResult;
    }
}
```

## 54.螺旋矩阵

```
class Solution {
    public List<Integer> spiralOrder(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        List<Integer> result = new ArrayList<>();
        int count = 1;
        int offset = 1;
        int startX = 0;
        int startY = 0;
        int loop = Math.min(m, n) / 2;
        int i = 0;
        int j = 0;

        while(loop > 0){
            i = startX;
            j = startY;

            for(j = startY; j < n - offset;j++){
                result.add(matrix[i][j]);
            }

            for(i = startX; i < m - offset;i++){
                result.add(matrix[i][j]);
            }

            for(;j > startY; j--){
                result.add(matrix[i][j]);
            }

            for(;i > startX;i--){
                result.add(matrix[i][j]);
            }

            startX++;
            startY++;
            offset++;

            loop --;
        }

        if(result.size() == n * m){
            return result;
        }
        
        //添加
        if(m > n){
            for(i = startX; i < m - offset;i++){
                result.add(matrix[i][startY]);
            }
            result.add(matrix[i][startY]);
        }else if(m < n){
            for(j = startY; j < n - offset; j++){
                result.add(matrix[startX][j]);
            }
            result.add(matrix[startX][j]);        
        }else if(n % 2 == 1 && n == m){
            result.add(matrix[n/2][n/2]);
        }

        return result;

    }
}
```

## 剑指offer29.顺时针打印矩阵

```
class Solution {
    public int[] spiralOrder(int[][] matrix) {
        int m = matrix.length;
        if(m == 0){
            return new int[0];
        }else if(m == 1){
            return matrix[0];
        }

        int n = matrix[0].length;
        int[] result = new int[m*n];

        int index = 0;  
        int offset = 1;
        int startX = 0;
        int startY = 0;
        int i = 0;
        int j = 0;
        int loop = Math.min(m, n) / 2;

        while(loop > 0){
            i = startX;
            j = startY;

            for(j = startY; j < n - offset; j ++){
                result[index++] = matrix[i][j];
            }

            for(i = startX; i < m - offset; i ++){
                result[index++] = matrix[i][j];
            }

            for(; j > startY; j--){
                result[index++] = matrix[i][j];
            }

            for(; i > startX; i--){
                result[index++] = matrix[i][j];
            }

            startX++;
            startY++;
            offset++;
            loop--;
        }

        if(index == m*n){
            System.out.println("here");
            return result;
        }

        if(m > n){
            for(i = startX; i < m - offset; i++){
                result[index++] = matrix[i][startY];
            }
            result[index++] = matrix[m - offset][startY];
        }else if(m < n){
            for(j = startY; j < n - offset; j++){
                result[index++] = matrix[startX][j];
            }
            result[index++] = matrix[startX][n - offset];
        }else if(m == n && m % 2 == 1){
            result[index++] = matrix[m / 2][m / 2];
        }

        return result;

    }
}
```

## 实现了
