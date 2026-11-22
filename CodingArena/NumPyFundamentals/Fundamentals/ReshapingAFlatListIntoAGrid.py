"""
    Reshaping a Flat List into a Grid
    Medium
    NumPy Fundamentals
    Write reshape_flat(values, rows, cols). If rows * cols equals
    len(values), reshape values into a rows by cols grid and return it as a
    list of lists. Otherwise, reshaping is impossible, so return None instead
    of crashing.

    Examples

    Arguments: [1, 2, 3, 4, 5, 6], 2, 3
    Returns: [[1, 2, 3], [4, 5, 6]]
    Explanation: Six values reshaped into 2 rows of 3 columns each.
    Constraints

    values is a list of numbers. 1 <= rows <= 50. 1 <= cols <= 50.
    len(values) does not have to equal rows*cols - when it doesn't, None is
    returned instead of a real grid.
"""
import numpy as np


def reshape_flat(values, rows, cols):
    """
    Return `values` as a rows x cols list of lists, or None if the sizes
    don't match.

    Time:  O(n) - building the array and the output list.
    Space: O(n)
    """
    # Step 1: list -> NumPy array.
    arr = np.asarray(values)

    # Step 2: reshape only works if the total number of cells stays the same.
    # rows * cols must equal len(values); otherwise arr.reshape raises
    # "ValueError: cannot reshape array of size 6 into shape (4,2)".
    # Checking FIRST avoids the crash (the "look before you leap" style).
    if rows * cols == len(values):
        # reshape fills ROW BY ROW (C order):
        #   [1, 2, 3, 4, 5, 6] -> [[1, 2, 3],
        #                          [4, 5, 6]]
        # .tolist() -> list of lists with plain Python numbers.
        return arr.reshape(rows, cols).tolist()

    # Step 3: sizes don't match -> None (as the task asks).
    return None


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        if rows * cols != len(values):
#            return None
#        return [values[r * cols:(r + 1) * cols] for r in range(rows)]
#
# 2. "Ask forgiveness" version - try the reshape, catch the error:
#        try:
#            return np.asarray(values).reshape(rows, cols).tolist()
#        except ValueError:
#            return None
#    Same result; the explicit check is clearer here.
#
# 3. reshape reminders:
#        arr.reshape(2, 3)    2 rows, 3 columns (filled row by row)
#        arr.reshape(-1, 3)   -1 = "work it out": rows = len / 3
#        arr.reshape(2, 3, order="F")  fill COLUMN by column instead:
#                             [[1, 3, 5], [2, 4, 6]]
#        arr.flatten() / arr.ravel()   back to 1-D
#        reshape returns a VIEW when it can: changing it can change arr.
#
# 4. Edge cases worth naming in an interview:
#      - rows * cols != len(values)  -> None (no crash)
#      - 1 row                       -> [[all values]]
#      - 1 column                    -> [[v1], [v2], ...]
#      - 1 value, 1 x 1              -> [[v]]
#      - empty list (rows, cols >= 1) -> rows*cols >= 1 != 0 -> None
#      - floats                      -> kept as floats
#
# 5. Related problems: LeetCode 566 (Reshape the Matrix) - the same task,
#    but returns the ORIGINAL matrix instead of None when it's impossible.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = list(range(2500))

    tests = [
        # (values, rows, cols, expected, description)
        ([1, 2, 3, 4, 5, 6], 2, 3, [[1, 2, 3], [4, 5, 6]], "given example"),
        ([1, 2, 3, 4, 5, 6], 3, 2, [[1, 2], [3, 4], [5, 6]], "3 x 2"),
        ([1, 2, 3, 4, 5, 6], 4, 2, None, "size mismatch -> None"),
        ([1, 2, 3], 1, 3, [[1, 2, 3]], "one row"),
        ([1, 2, 3], 3, 1, [[1], [2], [3]], "one column"),
        ([9], 1, 1, [[9]], "1 x 1"),
        ([], 1, 1, None, "empty list -> None"),
        ([1.5, 2.5, 3.5, 4.5], 2, 2, [[1.5, 2.5], [3.5, 4.5]], "floats"),
        (big, 50, 50, [big[r * 50:(r + 1) * 50] for r in range(50)], "max 50 x 50"),
    ]

    passed = 0
    for values, rows, cols, expected, desc in tests:
        got = reshape_flat(values, rows, cols)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if got is None or len(got) <= 3 else f"[{len(got)} x {len(got[0])} grid]"
        print(f"{status}: {desc:<24} ({rows}x{cols}) -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = reshape_flat([1, 2, 3, 4], 2, 2)
    assert type(out) is list and all(type(r) is list for r in out), "list of lists"
    assert all(type(v) is int for r in out for v in r), "plain Python ints"
    assert reshape_flat([1, 2, 3], 2, 2) is None, "None, not an error"
    print("PASS: list of lists / plain ints / no crash on mismatch")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()
    