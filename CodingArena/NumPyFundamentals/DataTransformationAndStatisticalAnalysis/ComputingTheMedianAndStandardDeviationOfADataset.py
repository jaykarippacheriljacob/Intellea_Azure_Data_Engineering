"""
    Computing the Median and Standard Deviation of a Dataset
    Medium
    Data Transformation & Statistical Analysis
    Write median_and_std(values). Return {"median": ..., "std": ...} for
    values, using np.median and np.std (standard deviation rounded to 4
    decimal places). If values is empty, return
    {"median": None, "std": None}.

    Examples

    Arguments: [1, 2, 3, 4, 5]
    Returns: {'median': 3.0, 'std': 1.4142}
    Explanation: Median of an odd count is the middle value, 3.0; the spread
    gives a std of about 1.4142.
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def median_and_std(values):
    """
    Return {"median", "std"} (std rounded to 4 decimals) as plain floats,
    or both None for an empty list.

    Time:  O(n log n) worst case for the median (partial sort), O(n) std.
    Space: O(n)
    """
    # Step 1: list -> array.
    arr = np.asarray(values)

    # Step 2: empty check FIRST. np.median([]) returns nan with a warning,
    # np.std([]) too - not an error, so it must be caught explicitly.
    # arr.size == 0 counts ALL elements (safer than `if not arr`, which
    # raises "truth value of an array is ambiguous").
    if arr.size == 0:
        return {"median": None, "std": None}

    # Step 3: median and population std.
    #   np.median: middle value (odd count) or mean of the two middle values
    #              (even count) -> always a float
    #   np.std:    population std (ddof=0, divides by n) - the task's
    #              example 1.4142 confirms ddof=0 (ddof=1 would give 1.5811)
    #
    # REVISION 1: float(...) - np.median / np.std return np.float64, not
    # plain Python floats ({'median': np.float64(3.0), ...}). float() gives
    # 3.0 and 1.4142 exactly as in the expected output.
    return {
        "median": float(np.median(arr)),
        "std": round(float(np.std(arr)), 4),
    }


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        import statistics
#        if not values:
#            return {"median": None, "std": None}
#        return {"median": float(statistics.median(values)),
#                "std": round(statistics.pstdev(values), 4)}
#    statistics.pstdev = population (ddof=0); statistics.stdev = sample.
#
# 2. ddof - a classic interview point:
#        np.std(x)          ddof=0  population std, divides by n     1.4142
#        np.std(x, ddof=1)  sample std, divides by n - 1             1.5811
#        pandas Series.std() uses ddof=1 by DEFAULT - so df["c"].std() and
#        np.std(df["c"]) give different numbers.
#
# 3. Median vs mean: the median ignores outliers.
#        [1, 2, 3, 4, 100] -> mean 22.0, median 3.0
#
# 4. NaN handling: np.median / np.std return nan if any value is NaN.
#        np.nanmedian(arr), np.nanstd(arr) skip NaN values.
#
# 5. Edge cases worth naming in an interview (tested):
#      - empty list       -> both None
#      - one value        -> median = the value, std = 0.0
#      - even count       -> median = mean of the two middle values
#      - all equal        -> std = 0.0
#      - unsorted input   -> median still correct (np.median sorts)
#      - negative values  -> work as normal


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = list(range(1, 1001))

    tests = [
        # (values, expected, description)
        ([1, 2, 3, 4, 5], {'median': 3.0, 'std': 1.4142}, "given example"),
        ([], {'median': None, 'std': None}, "empty list"),
        ([7], {'median': 7.0, 'std': 0.0}, "one value"),
        ([1, 2, 3, 4], {'median': 2.5, 'std': 1.118}, "even count"),
        ([5, 5, 5], {'median': 5.0, 'std': 0.0}, "all equal"),
        ([9, 1, 5], {'median': 5.0, 'std': 3.266}, "unsorted input"),
        ([-3, -1, 4], {'median': -1.0, 'std': 2.9439}, "negative values"),
        ([1, 2, 3, 4, 100], {'median': 3.0, 'std': 39.0128}, "outlier"),
        (big, {'median': 500.5, 'std': 288.6750}, "max size n=1000"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = median_and_std(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 5 else f"[{len(values)} values]"
        print(f"{status}: {desc:<18} {shown} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = median_and_std([1, 2, 3, 4, 5])
    assert type(out["median"]) is float and type(out["std"]) is float, "plain floats"
    assert list(out) == ["median", "std"], "key order"
    print("PASS: plain float types / key order")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()