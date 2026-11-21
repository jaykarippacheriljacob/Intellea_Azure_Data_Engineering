"""
    Summing and Averaging a List of Numbers
    Easy
    NumPy Fundamentals
    Write total_and_average(values). Build a numpy array from values and return {"total": ..., "average": ...} for it. If values is empty, return {"total": 0, "average": 0.0} instead of crashing.

    Examples

    Arguments: [10, 20, 30]
    Returns: {'total': 60, 'average': 20.0}
    Explanation: The three values sum to 60 and average to 20.0.
    Constraints

    values is a list of ints/floats. 0 <= len(values) <= 1000
"""
import numpy as np


def total_and_average(values):
    """
    Return {"total": sum of values, "average": mean of values}.
    An empty input gives {"total": 0, "average": 0.0}.

    Time:  O(n)  - one pass for the sum (the mean reuses it).
    Space: O(n)  - for the numpy array.
    """
    # Empty input: np.mean([]) does not crash, but it prints a
    # "Mean of empty slice" RuntimeWarning and returns nan, which is not the
    # 0.0 the task wants. So handle it up front. len() is used instead of
    # `if not values` because `not` raises on multi-element numpy arrays.
    if len(values) == 0:
        return {"total": 0, "average": 0.0}

    # Build the numpy array, as the task asks.
    arr = np.asarray(values)

    # Sum once, then compute the average from it instead of calling
    # arr.mean(), which would add everything up a second time.
    total = arr.sum()
    average = total / arr.size

    # .item() turns a numpy scalar into the matching Python type:
    #   np.int64(60)    -> 60    (int, so the output shows 60, not 60.0)
    #   np.float64(6.5) -> 6.5   (float, when the input contains floats)
    # This matters because the expected output is {'total': 60, ...} and
    # printing np.int64 in numpy 2.x shows "np.int64(60)".
    # float() makes sure the average is always a float (20.0, not 20).
    return {"total": total.item(), "average": float(average)}


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Simplest numpy version (also accepted, sums twice):
#        arr = np.array(values)
#        return {"total": arr.sum().item(), "average": float(arr.mean())}
#
# 2. Pure-Python version (no numpy):
#        total = sum(values)
#        return {"total": total, "average": total / len(values)}
#    (after the same empty check). Python's `/` always gives a float.
#
# 3. Common mistakes on this problem:
#      - Calling np.mean on an empty list -> nan plus a warning, not 0.0.
#      - Dividing by len(values) without the empty check ->
#        ZeroDivisionError in pure Python.
#      - Returning np.int64 / np.float64 instead of plain int / float.
#        `==` still matches, but strict judges compare types or printed
#        output.
#      - Returning float(total) -> 60.0 instead of the expected 60.
#
# 4. Floating-point accuracy: sum([0.1] * 10) gives 0.9999999999999999 in
#    plain Python. numpy's sum uses pairwise summation, which is more
#    accurate for long float lists; math.fsum(values) is exactly rounded.
#
# 5. Follow-up: "running average" / "moving average" -> np.cumsum(arr)
#    gives prefix sums, so the average of any window values[i:j] is
#    (prefix[j] - prefix[i]) / (j - i) in O(1). This is the prefix-sum
#    pattern behind LeetCode 303 (Range Sum Query), 643 (Maximum Average
#    Subarray I) and 1480 (Running Sum of 1d Array).
#
# 6. Follow-up: average without overflow. In languages with fixed-width
#    ints, (a + b) / 2 can overflow; use a + (b - a) / 2. Python ints are
#    unbounded, and numpy int64 only overflows above ~9.2e18.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    tests = [
        # (input, expected, description)
        ([10, 20, 30], {"total": 60, "average": 20.0}, "given example"),
        ([], {"total": 0, "average": 0.0}, "empty list"),
        ([7], {"total": 7, "average": 7.0}, "single element"),
        ([1, 2], {"total": 3, "average": 1.5}, "non-integer average"),
        ([-5, 5], {"total": 0, "average": 0.0}, "cancels to zero"),
        ([-1, -2, -3], {"total": -6, "average": -2.0}, "all negative"),
        ([1.5, 2.5], {"total": 4.0, "average": 2.0}, "floats"),
        ([1, 2.5], {"total": 3.5, "average": 1.75}, "mixed int/float"),
        ([0, 0, 0], {"total": 0, "average": 0.0}, "all zeros"),
        (list(range(1, 1001)), {"total": 500500, "average": 500.5},
         "max size n=1000"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = total_and_average(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 8 else f"[{len(values)} items]"
        print(f"{status}: {desc:<22} total_and_average({shown}) -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on types, not just values
    out = total_and_average([10, 20, 30])
    assert type(out["total"]) is int, "int input -> plain int total"
    assert type(out["average"]) is float, "average is a plain float"
    assert repr(out) == "{'total': 60, 'average': 20.0}", "matches output"
    assert type(total_and_average([1.5])["total"]) is float, "float total"
    empty = total_and_average([])
    assert type(empty["total"]) is int and type(empty["average"]) is float
    assert total_and_average(np.array([1, 2, 3])) == {"total": 6,
                                                      "average": 2.0}
    print("PASS: return types / exact printed output / numpy input")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()