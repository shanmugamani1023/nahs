# ── DSA 01: Arrays ────────────────────────────────────────
# Concept: Array = list of items stored in sequence
# Python list IS an array

# ── 1. Basic operations & complexity ──────────────────────
nums = [10, 20, 30, 40, 50]

print("── Basic Operations ──")
print(f"Access index 2   : {nums[2]}")        # O(1) — instant
print(f"Length           : {len(nums)}")       # O(1) — instant
nums.append(60)                                # O(1) — add to end
print(f"After append     : {nums}")
nums.pop()                                     # O(1) — remove from end
nums.insert(0, 5)                              # O(n) — shift all elements
print(f"After insert at 0: {nums}")
nums.pop(0)                                    # O(n) — shift all elements
print(f"After pop(0)     : {nums}")

# ── 2. Two Pointer technique ───────────────────────────────
# Problem: Check if array is palindrome
# [1, 2, 3, 2, 1] → True
# [1, 2, 3, 4, 5] → False

print("\n── Two Pointer: Palindrome Check ──")

def is_palindrome(arr):
    left  = 0
    right = len(arr) - 1

    while left < right:
        if arr[left] != arr[right]:
            return False
        left  += 1
        right -= 1

    return True

print(is_palindrome([1, 2, 3, 2, 1]))   # True
print(is_palindrome([1, 2, 3, 4, 5]))   # False
print(is_palindrome([1, 2, 2, 1]))      # True

# ── 3. Sliding Window technique ───────────────────────────
# Problem: Find max sum of k consecutive elements
# nums = [2, 1, 5, 1, 3, 2], k = 3
# windows: [2,1,5]=8, [1,5,1]=7, [5,1,3]=9, [1,3,2]=6
# Answer: 9

print("\n── Sliding Window: Max Sum of K Elements ──")

def max_sum_k(nums, k):
    # build first window
    window_sum = sum(nums[:k])
    max_sum    = window_sum

    # slide the window
    for i in range(k, len(nums)):
        window_sum += nums[i]       # add new element on right
        window_sum -= nums[i - k]   # remove old element on left
        max_sum = max(max_sum, window_sum)

    return max_sum

print(max_sum_k([2, 1, 5, 1, 3, 2], k=3))   # 9
print(max_sum_k([1, 4, 2, 9, 7, 3], k=2))   # 16

# ── YOUR TURN ──────────────────────────────────────────────
# Problem: Find the second largest number in an array
# Input : [3, 1, 4, 1, 5, 9, 2, 6]
# Output: 6  (largest=9, second=6)
#
# Try solving below:

def second_largest(nums):
    pass  # write your solution here

print("\n── Your Turn ──")
print(second_largest([3, 1, 4, 1, 5, 9, 2, 6]))  # expected: 6
print(second_largest([10, 5, 10]))                 # expected: 5
