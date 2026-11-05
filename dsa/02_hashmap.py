# ── DSA 02: HashMap (Dictionary) ──────────────────────────
# HashMap = Python dict
# O(1) insert, O(1) lookup — fastest data structure for counting

# ── Problem: First Non-Repeating Character ─────────────────
# Input : "aabbcddeff"
# Output: "c"

def first_non_repeating(s):
    # step 1: count frequency
    d = {}
    for c in s:
        d[c] = d.get(c, 0) + 1

    # step 2: find first with count = 1
    for c in s:
        if d[c] == 1:
            return c
    return None

print(first_non_repeating("aabbcddeff"))  # c
print(first_non_repeating("aabb"))        # None
print(first_non_repeating("abcd"))        # a

# ── YOUR TURN ──────────────────────────────────────────────
# Problem: Given a list of numbers, find two numbers that add up to target
# Input : nums=[2, 7, 11, 15], target=9
# Output: [0, 1]  (nums[0] + nums[1] = 2 + 7 = 9)
#
# Brute force: O(n²) — nested loop
# HashMap way : O(n)  — one loop
#
# Hint: for each num, check if (target - num) exists in dict

def two_sum(nums, target):
    pass

print(two_sum([2, 7, 11, 15], target=9))   # [0, 1]
print(two_sum([3, 2, 4],      target=6))   # [1, 2]
