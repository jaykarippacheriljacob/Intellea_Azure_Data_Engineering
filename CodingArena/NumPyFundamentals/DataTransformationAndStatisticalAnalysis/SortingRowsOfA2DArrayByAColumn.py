"""
    Sorting Rows of a 2D Array by a Column
    Medium
    Data Transformation & Statistical Analysis
    Write sort_by_column(grid, col_index, descending). Sort the rows of grid
    by the values in column col_index, ascending unless descending is True,
    and return the sorted grid as a list of lists. Tied rows must keep their
    ORIGINAL relative order.

    Examples

    Arguments: [[3, 1], [1, 2], [2, 0]], 0, False
    Returns: [[1, 2], [2, 0], [3, 1]]
    Explanation: Sorted ascending by column 0: [[1, 2], [2, 0], [3, 1]].
    Constraints

    grid is a non-empty rectangular list of lists of numbers.
    0 <= col_index < number of columns in grid.
"""
import numpy as np


def sort_by_column(grid, col_index, descending):
    """
    Return the rows of `grid` sorted by column `col_index` (ascending, or
    descending if `descending`), ties in original order, as list of lists.

    Time:  O(n log n) - sorting n rows.
    Space: O(n * k)
    """
    # Step 1: list of lists -> 2-D array.
    arr = np.asarray(grid)

    # Step 2: arr[:, col_index] = the sort column (1-D).
    # argsort() returns the ROW POSITIONS in sorted order (not the values);
    # arr[positions] then reorders whole rows (fancy indexing).
    #
    # kind="stable": rows with EQUAL values keep their original order
    # (the task requires this). The default quicksort does not promise it.
    #
    # Descending: NEGATE the column instead of reversing.
    #   values [5, 9, 1] -> negated [-5, -9, -1] -> smallest first = -9, -5, -1
    #   -> original 9, 5, 1 = biggest first. Equal values stay equal, so ties
    #   keep their order.
    #
    # ATTEMPTS that failed:
    #   - for i in col_index: ...       -> TypeError: 'int' object is not
    #     iterable (col_index is ONE number, not a list)
    #   - argsort()[::-1]               -> reverses EVERYTHING, ties too
    #   - argsort(kind="stable")[::-1]  -> stable, then reversed again: ties
    #     still flipped
    #   - (-col).argsort()[::-1]        -> minus AND reverse cancel out:
    #     "descending" came back ascending
    if descending:
        arr = arr[(-arr[:, col_index]).argsort(kind="stable")]
    else:
        arr = arr[arr[:, col_index].argsort(kind="stable")]

    # Step 3: array -> list of lists of plain Python numbers.
    return arr.tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy) - sorted() is ALWAYS stable, and
#    reverse=True keeps ties in order too:
#        return sorted(grid, key=lambda r: r[col_index], reverse=descending)
#
# 2. Why [::-1] breaks ties (example, sort by column 0, descending):
#        rows                 [1,10] [3,20] [1,30] [3,40]
#        stable ascending     [1,10] [1,30] [3,20] [3,40]
#        then [::-1]          [3,40] [3,20] [1,30] [1,10]   ties swapped
#        negate + stable      [3,20] [3,40] [1,10] [1,30]   correct
#
# 3. Brackets matter: (-arr[:, c]).argsort()  negate FIRST, then sort.
#    -arr[:, c].argsort() would sort first and negate the POSITIONS.
#
# 4. Sorting by several columns: np.lexsort (the LAST key is the main one):
#        order = np.lexsort((arr[:, 1], arr[:, 0]))   # by col 0, then col 1
#    pandas: df.sort_values(["a", "b"], ascending=[False, True],
#            kind="stable")
#
# 5. Type note: np.asarray makes ONE dtype for the whole grid, so a grid
#    mixing ints and floats comes back all floats ([[1, 2.5]] ->
#    [[1.0, 2.5]]). 1 == 1.0, so value checks still pass. To keep the
#    exact original rows:  [grid[i] for i in order]
#
# 6. Edge cases worth naming in an interview (tested):
#      - ties, ascending and descending -> original order kept
#      - one row                        -> unchanged
#      - already sorted / reverse sorted
#      - negative values, floats
#      - sort by the LAST column
#
# 7. Related: top_n_by_column (pandas sort_values with kind="stable").


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    ties = [[1, 10], [3, 20], [1, 30], [3, 40]]
    big = [[i % 7, i] for i in range(300)]

    tests = [
        # (grid, col_index, descending, description)
        ([[3, 1], [1, 2], [2, 0]], 0, False, "given example"),
        ([[3, 1], [1, 2], [2, 0]], 0, True, "descending"),
        ([[3, 1], [1, 2], [2, 0]], 1, False, "by column 1"),
        (ties, 0, False, "ties, ascending"),
        (ties, 0, True, "ties, descending"),
        ([[5, 5]], 0, True, "one row"),
        ([[1, 0], [2, 0], [3, 0]], 0, False, "already sorted"),
        ([[-2, 1], [4, 2], [-7, 3]], 0, True, "negative values"),
        ([[1.5, 9], [0.5, 8], [1.5, 7]], 0, True, "floats with tie"),
        ([[1, 2, 9], [3, 4, 1], [5, 6, 5]], 2, False, "last column"),
        (big, 0, True, "300 rows, many ties"),
    ]

    passed = 0
    for grid, col, desc_flag, desc in tests:
        got = sort_by_column(grid, col, desc_flag)
        # Python's sorted() is stable in both directions -> reference answer
        expected = sorted(grid, key=lambda r: r[col], reverse=desc_flag)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 4 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<22} col={col} desc={desc_flag!s:<5} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # The exact example from the task
    assert sort_by_column([[3, 1], [1, 2], [2, 0]], 0, False) == [[1, 2], [2, 0], [3, 1]]
    out = sort_by_column(ties, 0, True)
    assert out == [[3, 20], [3, 40], [1, 10], [1, 30]], "ties keep order"
    assert type(out) is list and all(type(r) is list for r in out), "list of lists"
    print("PASS: task example / tie order / list of lists")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()