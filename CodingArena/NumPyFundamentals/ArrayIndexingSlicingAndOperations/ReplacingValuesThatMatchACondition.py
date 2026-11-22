"""
    Replacing Values That Match a Condition
    Medium
    Array Indexing, Slicing & Operations
    Write replace_if_negative(values, replacement). Use
    np.where(arr < 0, replacement, arr) to replace every NEGATIVE value in
    values with replacement, leaving every other value untouched, and return
    the result as a plain list.

    Examples

    Arguments: [-3, 5, -1, 8], 0
    Returns: [0, 5, 0, 8]
    Explanation: Both negative values (-3 and -1) become 0; 5 and 8 stay
    untouched.
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def replace_if_negative(values, replacement):
    """
    Return `values` with every negative number replaced by `replacement`,
    as a plain list.

    Time:  O(n)
    Space: O(n) - np.where builds a new array.
    """
    # Step 1: list -> array.
    arr = np.asarray(values)

    # Step 2: np.where(condition, if_true, if_false)
    #   arr < 0       -> True/False mask, one per value
    #   True  -> replacement,  False -> the original value
    # Vectorised: the whole array at once, no loop.
    #
    # REVISION 1: the first version looped and wrote into arr in place:
    #     for i in range(len(arr)):
    #         if arr[i] < 0:
    #             arr[i] = replacement
    # Problems:
    #   - not np.where, as the task asks
    #   - HIDDEN BUG: arr keeps its dtype. [-3, 5] is an INT array, so
    #     arr[i] = 0.5 is silently truncated to 0 -> [0, 5] instead of
    #     [0.5, 5.0]. np.where makes a NEW array whose dtype fits both
    #     the values and the replacement.
    result = np.where(arr < 0, replacement, arr)

    # Step 3: array -> plain list of Python numbers.
    return result.tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        return [replacement if v < 0 else v for v in values]
#
# 2. Boolean-mask assignment (in place) - same idea as the loop, same dtype
#    trap:
#        arr = np.asarray(values, dtype=float)   # float avoids truncation
#        arr[arr < 0] = replacement
#
# 3. Related NumPy tools:
#        np.clip(arr, 0, None)        negatives -> 0 (only for replacing
#                                     with the boundary itself)
#        np.maximum(arr, 0)           same as above
#        np.where(arr < 0, -arr, arr) or np.abs(arr) -> absolute values
#        np.select([c1, c2], [v1, v2], default=arr)  several conditions
#
# 4. dtype rules (tested):
#        ints + int replacement     -> ints     [-3, 5], 0     -> [0, 5]
#        ints + float replacement   -> floats   [-3, 5], 0.5   -> [0.5, 5.0]
#        floats + int replacement   -> floats   [1.5, -2.5], 0 -> [1.5, 0.0]
#
# 5. Edge cases worth naming in an interview:
#      - empty list            -> []
#      - no negatives          -> unchanged
#      - all negative          -> all replacement
#      - zero is NOT negative  -> 0 stays 0 (condition is < 0, not <= 0)
#      - negative replacement  -> allowed, e.g. -99 as a marker
#
# 6. Related: bucket_by_condition (np.where with labels), NumPy
#    Fundamentals E3 (replace values below a threshold).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [i - 500 for i in range(1000)]

    tests = [
        # (values, replacement, expected, description)
        ([-3, 5, -1, 8], 0, [0, 5, 0, 8], "given example"),
        ([], 0, [], "empty list"),
        ([1, 2, 3], 0, [1, 2, 3], "no negatives"),
        ([-1, -2], 7, [7, 7], "all negative"),
        ([0, -1], 9, [0, 9], "zero is not negative"),
        ([-3, 5], 0.5, [0.5, 5.0], "float replacement (loop bug)"),
        ([1.5, -2.5], 0, [1.5, 0.0], "float values"),
        ([-3, 5], -99, [-99, 5], "negative replacement"),
        (big, 0, [max(v, 0) for v in big], "max size n=1000"),
    ]

    passed = 0
    for values, replacement, expected, desc in tests:
        got = replace_if_negative(values, replacement)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 5 else f"[{len(got)} values]"
        print(f"{status}: {desc:<30} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [-3, 5]
    out = replace_if_negative(data, 0)
    assert type(out) is list and all(type(v) is int for v in out), "plain ints"
    assert data == [-3, 5], "input unchanged"
    print("PASS: plain list of ints / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()