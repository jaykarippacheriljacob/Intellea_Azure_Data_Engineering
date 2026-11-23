"""
    Detecting Outliers Using the IQR Method
    Hard
    Data Transformation & Statistical Analysis
    Write iqr_outliers(values). Compute the 25th and 75th percentiles of
    values, derive the interquartile range (IQR = Q3 - Q1), and return every
    value falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR as a plain list,
    in original order. With fewer than 4 values there isn't enough data to
    compute meaningful quartiles, so return an empty list instead.

    Examples

    Arguments: [1, 2, 2, 3, 2, 90]
    Returns: [90.0]
    Explanation: The single extreme value, 90, is far enough outside the
    typical spread to be flagged.
    Constraints

    values is a list of numbers. Fewer than 4 values always returns an empty
    list, regardless of content. 0 <= len(values) <= 1000
"""
import numpy as np


def iqr_outliers(values):
    """
    Return the values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR], in original
    order, as a plain list of floats. Fewer than 4 values -> [].

    Time:  O(n log n) - percentiles need a (partial) sort.
    Space: O(n)
    """
    # REVISION 1: guard first. With < 4 values the task wants [], and
    # np.percentile([]) would raise an error.
    if len(values) < 4:
        return []

    # REVISION 2: dtype=float - the expected output is [90.0], not [90].
    # With whole numbers the array would be int64 and return ints.
    arr = np.asarray(values, dtype=float)

    # Quartiles and IQR.
    #   Q1 = 25th percentile, Q3 = 75th percentile (NumPy's default
    #   "linear" interpolation between data points)
    #   IQR = Q3 - Q1 = the spread of the middle 50 % of the data
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    # The "fences": anything outside them is an outlier.
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Boolean mask: True for outliers. | = OR, brackets around each part.
    # FIRST ATTEMPT: np.where(mask, arr) -> ValueError. np.where takes 1
    # argument (positions) or 3 (pick a/b), never 2. To SELECT values, use
    # boolean indexing: arr[mask].
    mask = (arr < lower) | (arr > upper)

    # REVISION 3: .tolist() -> plain list (arr[mask] is a NumPy array).
    # Boolean indexing keeps the ORIGINAL order.
    return arr[mask].tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Worked example [1, 2, 2, 3, 2, 90]:
#        sorted: 1, 2, 2, 2, 3, 90
#        Q1 = 2.0, Q3 = 2.75, IQR = 0.75
#        fences: 2.0 - 1.125 = 0.875   and   2.75 + 1.125 = 3.875
#        outside: 90 -> [90.0]   (1 is above 0.875, so it is NOT an outlier)
#
# 2. np.where with 1, 2 or 3 arguments:
#        np.where(mask)           -> positions (tuple of arrays)
#        np.where(mask, a, b)     -> a where True, b where False
#        np.where(mask, a)        -> ValueError
#    Positions of the outliers: np.where(mask)[0].tolist()
#
# 3. Same idea in pandas:
#        q1, q3 = s.quantile(0.25), s.quantile(0.75)
#        s[(s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))]
#    Box plots use exactly these fences for their "whiskers".
#
# 4. Percentile methods differ: NumPy's default is "linear". Other tools
#    (Excel QUARTILE.EXC, some textbooks) use other methods and can give
#    slightly different Q1/Q3 on small data. np.percentile(arr, 25,
#    method="midpoint") etc. if a task asks for another one.
#
# 5. Why IQR and not mean +/- 3 std? The quartiles barely move when an
#    outlier is added; the mean and std are pulled towards it.
#
# 6. Edge cases worth naming in an interview (tested):
#      - fewer than 4 values      -> [] (even with extreme values)
#      - no outliers              -> []
#      - outliers on both sides   -> both, in original order
#      - all values equal         -> IQR = 0, nothing outside -> []
#      - IQR = 0 but one value differs -> that value IS an outlier
#      - negative values / floats -> work as normal


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [5000.0] + [float(i) for i in range(998)] + [-5000.0]   # spread data + 2 extremes

    tests = [
        # (values, expected, description)
        ([1, 2, 2, 3, 2, 90], [90.0], "given example"),
        ([], [], "empty list"),
        ([1, 1000, -1000], [], "fewer than 4 -> []"),
        ([1, 2, 3, 4, 5], [], "no outliers"),
        ([-100, 10, 11, 12, 13, 14, 200], [-100.0, 200.0], "both sides, order kept"),
        ([200, 10, 11, 12, 13, 14, -100], [200.0, -100.0], "original order"),
        ([5, 5, 5, 5], [], "all equal (IQR 0)"),
        ([5, 5, 5, 5, 5, 9], [9.0], "IQR 0, one differs"),
        ([1.5, 2.5, 2.0, 2.2, 30.5], [30.5], "floats"),
        (big, [5000.0, -5000.0], "1000 values"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = iqr_outliers(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 7 else f"[{len(values)} values]"
        print(f"{status}: {desc:<24} {shown} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = iqr_outliers([1, 2, 2, 3, 2, 90])
    assert type(out) is list and all(type(v) is float for v in out), "plain floats"
    print("PASS: plain list of floats")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()