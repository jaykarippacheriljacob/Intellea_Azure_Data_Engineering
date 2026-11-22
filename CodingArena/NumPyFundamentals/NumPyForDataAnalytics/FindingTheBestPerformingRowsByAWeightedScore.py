
"""
    Finding the Best-Performing Row by a Weighted Score
    Hard
    NumPy for Data Analytics
    Write best_weighted_row(grid, weights). Compute each row's weighted score
    with a dot product (row_values . weights), and return
    {"row_index": ..., "score": ...} for the row with the HIGHEST score
    (score rounded to 4 decimal places). When multiple rows tie for the
    highest score, return the FIRST such row (lowest index).

    Examples

    Arguments: [[8, 7, 9], [9, 9, 9], [5, 10, 10]], [0.5, 0.3, 0.2]
    Returns: {'row_index': 1, 'score': 9.0}
    Explanation: Row 1's balanced 9s score highest under these weights.
    Constraints

    grid is a non-empty rectangular list of lists of numbers, one row per
    candidate. weights has exactly as many numbers as each row has columns.
    1 <= len(grid) <= 500
"""
import numpy as np


def best_weighted_row(grid, weights):
    """
    Return {"row_index", "score"} for the row with the highest weighted
    score (rounded to 4 decimals); the first row wins ties.

    Time:  O(rows * cols) - one matrix-vector product.
    Space: O(rows * cols)
    """
    # Step 1: lists -> arrays.
    arr = np.asarray(grid)
    w = np.asarray(weights)

    # Step 2: ALL scores at once. arr @ w (same as np.dot(arr, w)) does a
    # dot product of every row with the weights:
    #     row [8, 7, 9] . [0.5, 0.3, 0.2] = 4.0 + 2.1 + 1.8 = 7.9
    # shape (rows, cols) @ (cols,) -> (rows,)  one score per row.
    #
    # REVISION 1 (tie bug): round BEFORE comparing. Decimals like 0.1 are
    # not exact in binary, so equal scores can differ in the last digit:
    #     [0, 0, 1] . [0.1, 0.2, 0.3] = 0.3
    #     [1, 1, 0] . [0.1, 0.2, 0.3] = 0.30000000000000004
    # Comparing raw floats picked row 1; it is really a tie -> row 0.
    scores = np.round(arr @ w, 4)

    # Step 3: np.argmax returns the FIRST position of the maximum, which is
    # exactly the tie rule ("lowest index wins").
    idx = int(np.argmax(scores))

    # REVISION 2: plain Python types and a rounded score. The first version
    # returned np.float64 and the unrounded score (e.g. 7.8999999999999995).
    return {"row_index": idx, "score": float(scores[idx])}


def best_weighted_row_loop(grid, weights):
    """
    MY VERSION (loop), fixed. Same result as best_weighted_row.

    My original:
        arr = np.asarray(grid)
        n = len(arr)
        score = np.dot(arr[0], weights)
        idx = 0
        for i in range(1, n):
            s = np.dot(arr[i], weights)
            if score < s:
                score = s
                idx = i
        return {"row_index": idx, "score": score}

    What was wrong:
      1. score not rounded to 4 decimals
      2. returned np.float64, not a plain float
      3. float-noise tie: 0.3 < 0.30000000000000004 is True, so a later row
         with the SAME real score replaced the first one
    Fixes: round each score BEFORE comparing, and float() at the end.
    The strict `<` was already right: it keeps the FIRST row on ties.
    """
    arr = np.asarray(grid)
    n = len(arr)
    score = round(float(np.dot(arr[0], weights)), 4)      # fix 1 + 2 + 3
    idx = 0
    for i in range(1, n):
        s = round(float(np.dot(arr[i], weights)), 4)      # round BEFORE comparing
        if score < s:                                     # strict: first row wins ties
            score = s
            idx = i
    return {"row_index": idx, "score": score}


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. The first (loop) version, for comparison:
#        score = np.dot(arr[0], weights); idx = 0
#        for i in range(1, n):
#            s = np.dot(arr[i], weights)
#            if score < s:            # strict "<" keeps the FIRST on ties
#                score, idx = s, i
#    The tie logic was right; it failed on float noise and the missing
#    rounding / float() conversion.
#
# 2. Pure-Python version (no NumPy):
#        scores = [round(sum(v * w for v, w in zip(row, weights)), 4)
#                  for row in grid]
#        best = max(scores)
#        idx = scores.index(best)            # .index -> first occurrence
#        return {"row_index": idx, "score": best}
#
# 3. Dot product reminders:
#        np.dot(a, b)  ==  a @ b  ==  (a * b).sum()   for 1-D a, b
#        2-D @ 1-D -> one dot product per row
#        Weighted average = (row @ w) / w.sum()  (here weights sum to 1)
#
# 4. argmax family:
#        np.argmax / np.argmin   FIRST position of max / min
#        np.argsort(-scores, kind="stable")[:k]   top-k rows, ties in order
#        np.max(scores)          the value itself, not the position
#
# 5. Float comparison rule of thumb: never compare floats with == or < when
#    a tie matters - round first, or use np.isclose(a, b).
#
# 6. Edge cases worth naming in an interview (tested):
#      - exact tie                 -> first row
#      - float-noise tie           -> first row (thanks to rounding)
#      - one row                   -> index 0
#      - negative scores           -> the least negative wins
#      - integer weights           -> score still returned as float


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [[r % 50, (r * 7) % 50, (r * 3) % 50] for r in range(500)]
    big_scores = [round(sum(v * w for v, w in zip(row, [0.2, 0.5, 0.3])), 4) for row in big]

    tests = [
        # (grid, weights, expected, description)
        ([[8, 7, 9], [9, 9, 9], [5, 10, 10]], [0.5, 0.3, 0.2],
         {'row_index': 1, 'score': 9.0}, "given example"),
        ([[1, 2], [2, 1], [1, 2]], [1, 1],
         {'row_index': 0, 'score': 3.0}, "exact tie -> first"),
        ([[0, 0, 1], [1, 1, 0]], [0.1, 0.2, 0.3],
         {'row_index': 0, 'score': 0.3}, "float-noise tie -> first"),
        ([[4, 6]], [0.5, 0.5],
         {'row_index': 0, 'score': 5.0}, "one row"),
        ([[-5, -1], [-2, -2]], [1, 1],
         {'row_index': 1, 'score': -4.0}, "negative scores"),
        ([[1, 2, 3], [3, 2, 1]], [1, 2, 3],
         {'row_index': 0, 'score': 14.0}, "integer weights"),
        ([[1, 1, 1], [1, 1, 2]], [1 / 3, 1 / 3, 1 / 3],
         {'row_index': 1, 'score': 1.3333}, "score rounded to 4"),
        (big, [0.2, 0.5, 0.3],
         {'row_index': big_scores.index(max(big_scores)), 'score': max(big_scores)},
         "500 rows"),
    ]

    passed = 0
    for grid, weights, expected, desc in tests:
        got = best_weighted_row(grid, weights)
        got_loop = best_weighted_row_loop(grid, weights)
        ok = got == expected and got_loop == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<26} -> {got}  (loop: {got_loop})")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    for f in (best_weighted_row, best_weighted_row_loop):
        out = f([[8, 7, 9], [9, 9, 9]], [0.5, 0.3, 0.2])
        assert type(out["row_index"]) is int and type(out["score"]) is float, "plain types"
    print("PASS: plain int / float types (both versions)")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()
