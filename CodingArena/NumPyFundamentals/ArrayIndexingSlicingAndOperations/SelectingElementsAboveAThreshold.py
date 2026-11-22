"""
    Selecting Elements Above a Threshold
    Easy
    Array Indexing, Slicing & Operations
    Write above_threshold(values, threshold). Use boolean indexing
    (arr[arr > threshold]) to return only the elements of values that are
    STRICTLY greater than threshold, in their original order.

    Examples

    Arguments: [3, 8, 1, 9], 5
    Returns: [8, 9]
    Explanation: Only 8 and 9 clear a threshold of 5.
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def above_threshold(values, threshold):
    """
    Return the values strictly greater than `threshold`, in original order,
    as a plain list.

    Time:  O(n)
    Space: O(n)
    """
    # Step 1: list -> array.
    arr = np.asarray(values)

    # Step 2: BOOLEAN INDEXING (a mask).
    #   arr > threshold  -> [False, True, False, True]  one True/False each
    #   arr[mask]        -> keeps only the True positions, in order
    # ">" is STRICT: a value equal to threshold is left out.
    # The result is a new 1-D array (a copy).
    #
    # Step 3: .tolist() -> plain list of Python numbers.
    return arr[arr > threshold].tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        return [v for v in values if v > threshold]
#
# 2. Combining conditions (same rules as pandas filters):
#        arr[(arr > 2) & (arr < 9)]     AND - brackets around each part
#        arr[(arr < 2) | (arr > 8)]     OR
#        arr[~(arr > 5)]                NOT
#    Using `and` / `or` raises "truth value of an array is ambiguous".
#
# 3. Related tools:
#        (arr > t).sum()        HOW MANY values pass (True counts as 1)
#        np.where(arr > t)[0]   the POSITIONS that pass
#        np.count_nonzero(arr > t)  same as .sum()
#        arr[arr > t] = 0       boolean mask ASSIGNMENT (changes arr)
#
# 4. Same idea in pandas: df[df["col"] > t] (filter_by_condition).
#
# 5. Edge cases worth naming in an interview (tested):
#      - empty list            -> []
#      - value == threshold    -> excluded (strict)
#      - nothing passes        -> []
#      - everything passes     -> the whole list
#      - negative threshold    -> works as normal
#      - floats                -> kept as floats


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = list(range(1000))

    tests = [
        # (values, threshold, expected, description)
        ([3, 8, 1, 9], 5, [8, 9], "given example"),
        ([], 5, [], "empty list"),
        ([5, 6, 5], 5, [6], "equal is excluded"),
        ([1, 2, 3], 10, [], "nothing passes"),
        ([7, 8, 9], 0, [7, 8, 9], "everything passes"),
        ([-5, -1, 2], -2, [-1, 2], "negative threshold"),
        ([1.5, 2.5, 0.5], 1.0, [1.5, 2.5], "floats"),
        ([9, 1, 8, 2, 7], 5, [9, 8, 7], "original order kept"),
        (big, 996, [997, 998, 999], "max size n=1000"),
    ]

    passed = 0
    for values, threshold, expected, desc in tests:
        got = above_threshold(values, threshold)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 5 else f"[{len(values)} values]"
        print(f"{status}: {desc:<22} {shown} > {threshold} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = above_threshold([3, 8, 1, 9], 5)
    assert type(out) is list and all(type(v) is int for v in out), "plain ints"
    print("PASS: plain list of ints")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()