"""
    Fancy Indexing to Reorder and Select Rows
    Hard
    Array Indexing, Slicing & Operations
    Write select_rows_by_index(grid, indices). Use fancy indexing
    (arr[indices]) to build a new list of rows taken from grid in EXACTLY the
    order given by indices - an index may repeat, and negative indices count
    from the end, just like plain Python indexing.

    Examples

    Arguments: [[10, 11], [20, 21], [30, 31]], [2, 0]
    Returns: [[30, 31], [10, 11]]
    Explanation: Row 2 first, then row 0: [[30, 31], [10, 11]].
    Constraints

    grid is a non-empty rectangular list of lists of numbers. indices is a
    list of valid row positions into grid (which may repeat a row or use
    negative indices); its own order controls the output order.
"""
import numpy as np


def select_rows_by_index(grid, indices):
    """
    Return the rows of `grid` at `indices`, in that order, as a list of
    lists.

    Time:  O(len(indices) * cols)
    Space: O(len(indices) * cols) - fancy indexing makes a COPY.
    """
    # Step 1: list of lists -> 2-D array.
    arr = np.asarray(grid)

    # Step 2: FANCY INDEXING - pass a LIST of positions in one go.
    #   arr[[2, 0]]      -> rows 2 then 0
    #   arr[[0, 0]]      -> row 0 twice (repeats allowed)
    #   arr[[-1]]        -> last row (negative counts from the end)
    # The output order follows `indices`, not the grid's order.
    #
    # REVISION 1 (bug fix): the first version looped and appended arr[i]:
    #     [array([30, 31]), array([10, 11])]
    # - a list of NumPy ARRAYS, not lists. Comparing it with
    #   [[30, 31], [10, 11]] raises "truth value of an array ... is
    #   ambiguous". It also didn't use fancy indexing as the task asks.
    # .tolist() on the result gives plain nested lists of Python ints.
    return arr[indices].tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. The loop version, fixed (works, but not "fancy indexing"):
#        return [arr[i].tolist() for i in indices]
#    Pure Python, no NumPy:
#        return [list(grid[i]) for i in indices]
#
# 2. Fancy indexing vs slicing (interview point):
#        arr[0:2]       SLICE  -> a VIEW, rows in grid order, no repeats
#        arr[[2, 0]]    FANCY  -> a COPY, any order, repeats allowed
#    Changing a view changes the original; changing a fancy result doesn't.
#
# 3. More fancy indexing:
#        arr[:, [1, 0]]            reorder / pick COLUMNS
#        arr[[0, 2], [1, 0]]       PAIRS: elements (0,1) and (2,0) -> 1-D
#        arr[np.ix_([0, 2], [1, 0])]   rows 0,2 x cols 1,0 -> 2x2 block
#        arr[arr[:, 0] > 15]       boolean mask: rows where col 0 > 15
#
# 4. Edge cases worth naming in an interview (tested):
#      - repeated index [0, 0]   -> the row twice
#      - negative index [-1]     -> last row
#      - empty indices []        -> []
#      - same order as grid      -> a copy of the grid
#      - out-of-range index      -> IndexError (ruled out by constraints)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    g = [[10, 11], [20, 21], [30, 31]]
    big = [[r, r * 2] for r in range(100)]

    tests = [
        # (grid, indices, expected, description)
        (g, [2, 0], [[30, 31], [10, 11]], "given example"),
        (g, [0, 0, 1], [[10, 11], [10, 11], [20, 21]], "repeated index"),
        (g, [-1], [[30, 31]], "negative index"),
        (g, [-1, -3], [[30, 31], [10, 11]], "negatives, reordered"),
        (g, [0, 1, 2], g, "same order = copy"),
        (g, [], [], "empty indices"),
        ([[1.5, 2.5], [3.5, 4.5]], [1], [[3.5, 4.5]], "floats"),
        ([[7, 8, 9]], [0, 0], [[7, 8, 9], [7, 8, 9]], "one-row grid"),
        (big, [99, 0, 50], [[99, 198], [0, 0], [50, 100]], "100-row grid"),
    ]

    passed = 0
    for grid, indices, expected, desc in tests:
        got = select_rows_by_index(grid, indices)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<22} {indices} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = select_rows_by_index(g, [2, 0])
    assert type(out) is list and all(type(r) is list for r in out), "list of lists"
    assert all(type(v) is int for r in out for v in r), "plain Python ints"
    out[0][0] = 999
    assert g[2][0] == 30, "input unchanged (fancy indexing copies)"
    print("PASS: list of lists / plain ints / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()