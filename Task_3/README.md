### Python Algorithm Practice – Internship Task 3

This repository contains solutions to multiple Python algorithmic problems completed as part of Internship Task 3.
The task focuses on strengthening problem-solving skills, understanding algorithmic logic, and applying core Python concepts using clean and readable code.

All solutions follow a class-based structure commonly used in coding platforms such as LeetCode.

### Objective

---Improve logical thinking and algorithm design
---Practice Python fundamentals through real interview-style problems
---Write efficient and readable solutions

### Technologies & Concepts Used
-->Python 3
-->Jupyter Notebook
-->Core programming concepts:
-->Arrays & lists
-->Nested loops
-->Conditional logic
-->Bit manipulation
-->Time interval checks

### Problems Implemented
1️. Maximum Product of Two Elements in an Array

Finds two different indices such that:
 (nums[i] - 1) * (nums[j] - 1) is maximized.
  ## Approach:
  -->Sort the array
  -->Use the two largest values
  -->Demonstrates sorting and optimization logic
  
2️. Count Number of Teams
Counts the number of valid teams of 3 soldiers where:
--->Ratings are strictly increasing or decreasing
--->Uses triple nested loops to validate all combinations
--->Demonstrates brute-force logic and condition checking

3️. Number of Students Doing Homework at a Given Time
Counts how many students are active at a specific query time
Checks if:
--->startTime[i] ≤ queryTime ≤ endTime[i]
--->Demonstrates interval logic and iteration

4️. Number of Steps to Reduce a Number to Zero
Reduces a number to zero using:
---->Divide by 2 (if even)
---->Subtract 1 (if odd)
--->Demonstrates while loops and conditional branching

5️. Counting Bits
Returns an array where each index stores the number of set bits (1s) in the binary representation of the index
Uses dynamic programming
## Efficient approach:
ans[i] = ans[i // 2] + (i % 2)
--->Demonstrates bit manipulation and optimization
