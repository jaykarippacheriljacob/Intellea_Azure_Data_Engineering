"""
    Extracting a Sub-Grid via 2D Slicing
    Medium
    Array Indexing, Slicing & Operations
    Write sub_grid(grid, row_start, row_end, col_start, col_end). Use 2D
    slicing (arr[row_start:row_end, col_start:col_end]) to extract and return
    the corresponding sub-grid as a list of lists.

    Examples

    Arguments: [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 0, 2, 1, 3
    Returns: [[2, 3], [5, 6]]
    Explanation: Rows 0-1 and columns 1-2: [[2, 3], [5, 6]].
    Constraints

    grid is a non-empty rectangular list of lists of numbers.
    0 <= row_start <= row_end <= len(grid).
    0 <= col_start <= col_end <= len(grid[0]).
"""
import numpy as np


def sub_grid(grid, row_start, row_end, col_start, col_end):
    """
    Return rows [row_start, row_end) and columns [col_start, col_end) of
    `grid` as a list of lists (ends excluded, like Python slicing).

    Time:  O(r * c) for the r x c result (tolist copies it).
    Space: O(r * c)
    """
    # Step 1: list of lists -> 2-D array.
    arr = np.asarray(grid)

    # Step 2: 2-D SLICING - arr[rows, cols] in ONE step.
    #   row_start:row_end  -> rows row_start .. row_end-1  (end excluded)
    #   col_start:col_end  -> cols col_start .. col_end-1
    # The slice is a VIEW: no copying until .tolist().
    #
    # REVISION 1: the first version built the result with nested loops into
    # np.zeros((rows, cols)). Problems:
    #   - np.zeros makes a FLOAT array, so ints came back as floats:
    #     [[2.0, 3.0], [5.0, 6.0]] instead of [[2, 3], [5, 6]]
    #   - big ints lost precision: 100000000000000001 -> 1e+17
    #   - the task asks for 2-D slicing, which replaces both loops
    piece = arr[row_start:row_end, col_start:col_end]

    # Step 3: keep the original dtype, convert to plain nested lists.
    return piece.tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        return [row[col_start:col_end] for row in grid[row_start:row_end]]
#
# 2. 2-D slicing vs plain lists (interview point):
#        arr[0:2, 1:3]     NumPy: rows AND columns in one bracket
#        grid[0:2][1:3]    plain list: WRONG - slices rows twice, not
#                          rows then columns
#
# 3. Slicing reminders:
#        arr[1]          row 1 (1-D)
#        arr[:, 1]       column 1 (1-D)
#        arr[:, 1:2]     column 1, kept 2-D
#        arr[::2, ::2]   every other row and column (step)
#        arr[-2:, -2:]   bottom-right 2x2 block
#
# 4. View vs copy:
#        piece = arr[0:2, 1:3]  -> a VIEW: piece[0, 0] = 99 changes arr too
#        piece.copy()           -> independent copy
#        .tolist() always builds new Python lists, so the grid is safe here.
#
# 5. Why np.zeros caused floats: np.zeros, np.ones and np.empty default to
#    float64. Use np.zeros(shape, dtype=arr.dtype) to match the input.
#
# 6. Edge cases worth naming in an interview (tested):
#      - row_start == row_end -> no rows -> []
#      - col_start == col_end -> rows with no columns -> [[], [], ...]
#      - whole grid           -> a copy of grid
#      - single cell          -> [[value]]
#      - floats               -> kept as floats


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    tests = [
        # (grid, rs, re, cs, ce, expected, description)
        (g, 0, 2, 1, 3, [[2, 3], [5, 6]], "given example"),
        (g, 0, 3, 0, 3, g, "whole grid"),
        (g, 1, 2, 1, 2, [[5]], "single cell"),
        (g, 2, 3, 0, 3, [[7, 8, 9]], "last row"),
        (g, 0, 3, 2, 3, [[3], [6], [9]], "last column"),
        (g, 1, 1, 0, 3, [], "no rows -> []"),
        (g, 0, 2, 1, 1, [[], []], "no columns -> [[], []]"),
        ([[1.5, 2.5], [3.5, 4.5]], 0, 2, 0, 1, [[1.5], [3.5]], "floats"),
        ([[10 ** 17 + 1]], 0, 1, 0, 1, [[10 ** 17 + 1]], "big int exact"),
    ]

    passed = 0
    for grid, rs, re_, cs, ce, expected, desc in tests:
        got = sub_grid(grid, rs, re_, cs, ce)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<24} [{rs}:{re_}, {cs}:{ce}] -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = sub_grid(g, 0, 2, 1, 3)
    assert all(type(v) is int for r in out for v in r), "ints stay ints"
    out[0][0] = 99
    assert g[0][1] == 2, "input unchanged"
    print("PASS: ints stay ints / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()