"""
    Computing a Cumulative Running Total
    Easy
    NumPy for Data Analytics
    Write running_total(values). Use np.cumsum(values) to build the running
    total at every position, and return it as a plain list.

    Examples

    Arguments: [10, 20, 30]
    Returns: [10, 30, 60]
    Explanation: Running totals: [10, 30, 60].
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def running_total(values):
    """
    Return the running total at every position as a plain list.

    Time:  O(n)
    Space: O(n)
    """
    # Step 1: list -> array.
    arr = np.asarray(values)

    # Step 2: np.cumsum = CUMULATIVE SUM. Position i holds the sum of
    # everything from position 0 up to and including i:
    #     [10, 20, 30] -> [10, 10+20, 10+20+30] = [10, 30, 60]
    # The output has the SAME length as the input; the last value is the
    # grand total (same as np.sum).
    #
    # Step 3: .tolist() -> plain Python list.
    return np.cumsum(arr).tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python versions (no NumPy):
#        from itertools import accumulate
#        return list(accumulate(values))
#    or a loop:
#        total, out = 0, []
#        for v in values:
#            total += v
#            out.append(total)
#        return out
#
# 2. The same idea elsewhere:
#        pandas:   df["col"].cumsum()           (Midstone 1 Q6: revenue
#                                                 building up over the day)
#        per group: df.groupby("g")["v"].cumsum()
#        SQL:      SUM(v) OVER (ORDER BY t)     (window function)
#
# 3. Other cumulative functions:
#        np.cumprod  running product    [1, 2, 3] -> [1, 2, 6]
#        np.maximum.accumulate  running max  [3, 1, 4] -> [3, 3, 4]
#        np.diff     the OPPOSITE: differences between neighbours
#                    np.diff([10, 30, 60]) -> [20, 30]
#
# 4. Prefix sums (interview pattern): with c = cumsum, the sum of
#    values[i..j] is c[j] - c[i-1] - any range sum in O(1).
#    LeetCode 1480 (Running Sum of 1d Array), 303 (Range Sum Query).
#
# 5. Edge cases worth naming in an interview (tested):
#      - empty list     -> []
#      - one value      -> [value]
#      - negatives      -> total can go down
#      - zeros          -> total stays the same
#      - floats         -> float results (may show tiny binary errors,
#                          e.g. 0.1 + 0.2 = 0.30000000000000004)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = list(range(1, 1001))

    tests = [
        # (values, expected, description)
        ([10, 20, 30], [10, 30, 60], "given example"),
        ([], [], "empty list"),
        ([7], [7], "one value"),
        ([5, -3, -4, 10], [5, 2, -2, 8], "negatives"),
        ([1, 0, 0, 2], [1, 1, 1, 3], "zeros"),
        ([0.5, 1.5, 2.0], [0.5, 2.0, 4.0], "floats"),
        (big, [i * (i + 1) // 2 for i in range(1, 1001)], "max size n=1000"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = running_total(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 4 else f"[{len(got)} values, last {got[-1]}]"
        print(f"{status}: {desc:<16} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = running_total([10, 20, 30])
    assert type(out) is list and all(type(v) is int for v in out), "plain ints"
    assert out[-1] == sum([10, 20, 30]), "last value = grand total"
    print("PASS: plain ints / last value = total")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()