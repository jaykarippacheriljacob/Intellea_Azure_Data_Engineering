"""
    Computing Column-wise Totals for a Dataset Grid
    Medium
    NumPy for Data Analytics
    Write column_totals(grid). Use arr.sum(axis=0) to add up each column of
    grid down its rows, and return the totals as a plain list.

    Examples

    Arguments: [[1, 2, 3], [4, 5, 6]]
    Returns: [5, 7, 9]
    Explanation: Column totals: [5, 7, 9].
    Constraints

    grid is a non-empty rectangular list of lists of numbers.
    1 <= len(grid) <= 500
"""
import numpy as np


def column_totals(grid):
    """
    Return the total of each column (summed down the rows) as a plain list.

    Time:  O(rows * cols)
    Space: O(rows * cols) for the array, O(cols) for the result.
    """
    # Step 1: list of lists -> 2-D array.
    arr = np.asarray(grid)

    # Step 2: axis=0 collapses the ROWS -> one total per COLUMN.
    #     [[1, 2, 3],
    #      [4, 5, 6]]   -> axis=0 -> [1+4, 2+5, 3+6] = [5, 7, 9]
    # Memory trick: the axis you name is the one that DISAPPEARS.
    #     shape (2, 3) --sum(axis=0)--> shape (3,)
    #
    # Step 3: .tolist() -> plain Python list.
    return arr.sum(axis=0).tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy) - zip(*grid) turns columns into tuples:
#        return [sum(col) for col in zip(*grid)]
#
# 2. axis cheat sheet for a 2 x 3 grid [[1, 2, 3], [4, 5, 6]]:
#        arr.sum()          -> 21          everything
#        arr.sum(axis=0)    -> [5, 7, 9]   per COLUMN (down the rows)
#        arr.sum(axis=1)    -> [6, 15]     per ROW (across the columns)
#    The same axis rule works for mean, max, min, std, argmax...
#        arr.mean(axis=0) -> column averages, arr.max(axis=1) -> row maxima
#
# 3. Same idea in pandas: df.sum() sums each COLUMN by default (axis=0);
#    df.sum(axis=1) sums each row.
#
# 4. keepdims=True keeps the result 2-D (shape (1, 3)) - handy for
#    broadcasting, e.g. arr / arr.sum(axis=0, keepdims=True) gives each
#    value's share of its column.
#
# 5. Edge cases worth naming in an interview (tested):
#      - one row        -> the row itself
#      - one column     -> [total]
#      - negatives      -> can cancel out to 0
#      - floats         -> float totals
#      - mixed int/float -> all floats (one dtype per array)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [[r, r * 2, 1] for r in range(500)]

    tests = [
        # (grid, expected, description)
        ([[1, 2, 3], [4, 5, 6]], [5, 7, 9], "given example"),
        ([[7, 8, 9]], [7, 8, 9], "one row"),
        ([[1], [2], [3]], [6], "one column"),
        ([[5, -2], [-5, 2]], [0, 0], "negatives cancel"),
        ([[0.5, 1.5], [1.0, 2.0]], [1.5, 3.5], "floats"),
        ([[1, 2.5], [3, 4]], [4.0, 6.5], "mixed int/float"),
        (big, [sum(range(500)), 2 * sum(range(500)), 500], "500 rows"),
    ]

    passed = 0
    for grid, expected, desc in tests:
        got = column_totals(grid)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = grid if len(grid) <= 3 else f"[{len(grid)} rows]"
        print(f"{status}: {desc:<18} {shown} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    g = [[1, 2, 3], [4, 5, 6]]
    out = column_totals(g)
    assert type(out) is list and all(type(v) is int for v in out), "plain ints"
    assert len(out) == len(g[0]), "one total per column"
    assert np.asarray(g).sum(axis=1).tolist() == [6, 15], "axis=1 would be rows"
    print("PASS: plain ints / one per column / axis=1 differs")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()