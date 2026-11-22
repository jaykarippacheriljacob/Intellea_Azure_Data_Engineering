"""
    Counting How Many Values Satisfy a Condition
    Easy
    NumPy for Data Analytics
    Write count_above(values, threshold). Count how many elements of values
    are STRICTLY greater than threshold, using boolean indexing, and return
    that count as a plain int.

    Examples

    Arguments: [3, 8, 1, 9, 5], 4
    Returns: 3
    Explanation: Three values (8, 9, 5) clear a threshold of 4.
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def count_above(values, threshold):
    """
    Return how many values are strictly greater than `threshold`, as a
    plain Python int.

    Time:  O(n)
    Space: O(n) - the mask and the selected values.
    """
    # Step 1: list -> array.
    arr = np.asarray(values)

    # Step 2: BOOLEAN INDEXING - the mask selects the values that pass:
    #     arr > threshold        -> [False, True, False, True, True]
    #     arr[arr > threshold]   -> [8, 9, 5]
    # .size = how many values were selected.
    #
    # REVISION 1: the first version returned (arr > threshold).sum().
    # The count was right (True = 1, False = 0), but:
    #   - it returns np.int64(3), not a plain int as the task asks
    #   - it counts with the mask directly instead of boolean INDEXING
    # int(...) turns the NumPy integer into a plain Python int.
    return int(arr[arr > threshold].size)


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Equivalent ways to count (all give 3 for the example):
#        int((arr > threshold).sum())        sum of the True/False mask
#        int(np.count_nonzero(arr > threshold))
#        len(arr[arr > threshold])           boolean indexing + len
#    Pure Python: sum(1 for v in values if v > threshold)
#
# 2. COUNT vs SUM (easy to mix up):
#        (arr > 4).sum()         -> 3    how MANY pass (sums True/False)
#        arr[arr > 4].sum()      -> 22   the TOTAL of the values that pass
#
# 3. Percentage instead of count:
#        (arr > threshold).mean() * 100    -> 60.0 (3 of 5)
#
# 4. Why int(): NumPy returns np.int64. It prints as 3 and == 3 is True,
#    but type checks, json.dumps and some graders treat it differently.
#
# 5. Edge cases worth naming in an interview (tested):
#      - empty list          -> 0
#      - value == threshold  -> not counted (strict)
#      - nothing passes      -> 0
#      - everything passes   -> len(values)
#      - negatives / floats  -> work as normal


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = list(range(1000))

    tests = [
        # (values, threshold, expected, description)
        ([3, 8, 1, 9, 5], 4, 3, "given example"),
        ([], 4, 0, "empty list"),
        ([4, 4, 5], 4, 1, "equal is not counted"),
        ([1, 2, 3], 10, 0, "nothing passes"),
        ([5, 6, 7], 0, 3, "everything passes"),
        ([-5, -1, 2], -2, 2, "negative threshold"),
        ([0.5, 1.5, 2.5], 1.0, 2, "floats"),
        (big, 899, 100, "max size n=1000"),
    ]

    passed = 0
    for values, threshold, expected, desc in tests:
        got = count_above(values, threshold)
        ok = got == expected and type(got) is int
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 5 else f"[{len(values)} values]"
        print(f"{status}: {desc:<22} {shown} > {threshold} -> {got!r}")
        if not ok:
            print(f"      expected {expected!r} (plain int)")

    print(f"\n{passed}/{len(tests)} value tests passed (incl. plain-int type check)")


if __name__ == "__main__":
    run_tests()