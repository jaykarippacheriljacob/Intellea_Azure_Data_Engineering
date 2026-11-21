"""
    Computing the Range of a Dataset
    Easy
    Data Transformation & Statistical Analysis
    Write data_range(values). Return max(values) - min(values) as a float. If values is empty, return None instead of crashing.

    Examples

    Arguments: [3, 9, 1, 7]
    Returns: 8.0
    Explanation: Max 9 minus min 1 is a range of 8.0.
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def data_range(values):
    """
    Return the range of `values` (largest value minus smallest) as a float,
    or None if `values` is empty.

    Time:  O(n)  - one pass for max, one pass for min.
    Space: O(n)  - for the numpy array copy of the input.
    """
    # REVISION 1 (bug fix): the original `float(max(values) - min(values))`
    # crashes on an empty list with "ValueError: max() arg is an empty
    # sequence". The task says to return None instead, so check first.
    # We use len(...) == 0 rather than `if not values` because `not` on a
    # numpy array with several elements raises an error ("truth value of an
    # array is ambiguous"), and len() works for lists, tuples and arrays.
    if len(values) == 0:
        return None

    # REVISION 2: actually use numpy (the file imports it but the original
    # never used it). np.asarray turns the list into an array.
    arr = np.asarray(values)

    # Range = max - min. arr.max() and arr.min() are fast C loops.
    # (np.ptp(arr) - "peak to peak" - computes exactly the same thing.)
    spread = arr.max() - arr.min()

    # float() turns the numpy scalar (e.g. np.int64(8)) into a plain Python
    # float, so the result is 8.0 as the example expects.
    return float(spread)


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no numpy) - perfectly fine for n <= 1000:
#        def data_range(values):
#            if not values:
#                return None
#            return float(max(values) - min(values))
#
# 2. Single-pass version (the classic interview answer when asked to find
#    min and max "in one loop"):
#        def data_range(values):
#            if not values:
#                return None
#            lo = hi = values[0]
#            for v in values[1:]:
#                if v < lo:
#                    lo = v
#                elif v > hi:
#                    hi = v
#            return float(hi - lo)
#    Still O(n), but reads the data once instead of twice. Comparing in
#    pairs gets this down to ~1.5n comparisons instead of 2n (a well-known
#    textbook trick).
#
# 3. numpy shortcut: np.ptp(values) ("peak to peak") returns max - min
#    directly. It still raises on empty input, so the None check is needed.
#
# 4. Edge cases worth naming in an interview:
#      - empty list          -> None (handled)
#      - one element         -> 0.0 (max == min)
#      - all equal values    -> 0.0
#      - negatives / floats  -> works as normal
#      - NaN in the data     -> numpy's max/min return nan, so the range is
#        nan. Use np.nanmax(arr) - np.nanmin(arr) to ignore NaNs.
#      - huge integers       -> Python ints never overflow, but numpy int64
#        can (beyond ~9.2e18). Not a concern under these constraints.
#
# 5. Related problems: LeetCode 1491 (Average Salary Excluding the Minimum
#    and Maximum), LeetCode 2733 (Neither Minimum nor Maximum),
#    LeetCode 908 (Smallest Range I), which is built on max - min.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    tests = [
        # (input, expected, description)
        ([3, 9, 1, 7], 8.0, "given example"),
        ([], None, "empty list -> None"),
        ([5], 0.0, "single element"),
        ([4, 4, 4, 4], 0.0, "all equal"),
        ([-10, -3, -7], 7.0, "all negative"),
        ([-5, 0, 5], 10.0, "negative and positive"),
        ([1.5, 2.25, 0.5], 1.75, "floats"),
        ([1, 2], 1.0, "two elements"),
        ([10, 9, 8, 7, 1], 9.0, "sorted descending"),
        (list(range(1000)), 999.0, "max size n=1000"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = data_range(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 8 else f"[{len(values)} items]"
        print(f"{status}: {desc:<24} data_range({shown}) -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = data_range([3, 9, 1, 7])
    assert type(out) is float, "should return a plain Python float"
    assert data_range((3, 9, 1)) == 8.0, "tuples should work too"
    assert data_range(np.array([3, 9, 1])) == 8.0, "numpy input works"
    assert data_range(np.array([])) is None, "empty numpy array -> None"
    print("PASS: return type / alternate input types")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()