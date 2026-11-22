"""
    Building an Evenly Spaced Sequence Between Two Bounds
    Hard
    NumPy Fundamentals
    Write evenly_spaced(start, stop, count). Use np.linspace(start, stop,
    count) to build count evenly spaced numbers from start to stop (inclusive
    of both ends), rounded to 6 decimal places, and return them as a plain
    list. If count is 0, return an empty list.

    Examples

    Arguments: 0, 10, 5
    Returns: [0.0, 2.5, 5.0, 7.5, 10.0]
    Explanation: Five evenly spaced points from 0 to 10:
    [0.0, 2.5, 5.0, 7.5, 10.0].
    Constraints

    start and stop are numbers; start may be greater than, equal to, or less
    than stop. 0 <= count <= 100
"""
import numpy as np


def evenly_spaced(start, stop, count):
    """
    Return `count` evenly spaced numbers from start to stop (both included),
    rounded to 6 decimals, as a plain Python list.

    Time:  O(count)
    Space: O(count)
    """
    # np.linspace(start, stop, count):
    #   - includes BOTH ends (endpoint=True by default)
    #   - step = (stop - start) / (count - 1)
    #   - count = 1 -> [start];  count = 0 -> empty array
    #   - works when start > stop (counts down) or start == stop
    arr = np.linspace(start, stop, count)

    # np.round(arr, 6): NumPy's rounding for a whole array.
    # REVISION 1: round(arr, 4) (Python's built-in) -> TypeError: numpy.ndarray
    #   doesn't define __round__. Use np.round(arr, n) or arr.round(n).
    # REVISION 2: np.round(arr) with no decimals rounds to WHOLE numbers ->
    #   2.5 became 2.0 and 7.5 became 8.0 (round-half-to-even). Pass 6.
    #
    # .tolist(): array -> plain list, np.float64 -> Python float.
    #
    # REVISION 3: removed `if count == 0: return None`. The task wants [] for
    #   count 0, and np.linspace(..., 0) already gives an empty array, so
    #   .tolist() returns [] with no special case. Keep the return type the
    #   same (always a list).
    return np.round(arr, 6).tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        if count == 0: return []
#        if count == 1: return [round(float(start), 6)]
#        step = (stop - start) / (count - 1)
#        return [round(start + i * step, 6) for i in range(count)]
#
# 2. linspace vs arange (classic interview question):
#        np.linspace(0, 10, 5)  -> [0, 2.5, 5, 7.5, 10]  give the COUNT,
#                                  stop INCLUDED
#        np.arange(0, 10, 2.5)  -> [0, 2.5, 5, 7.5]      give the STEP,
#                                  stop EXCLUDED, float steps can drift
#        np.linspace(0, 10, 5, endpoint=False) -> [0, 2, 4, 6, 8]
#
# 3. Why round at all? Floating-point steps are not exact:
#        np.linspace(0, 1, 4) -> [0.0, 0.333333333..., 0.666666666..., 1.0]
#        Rounding to 6 decimals gives stable values to compare.
#
# 4. Rounding rules:
#        round(x, n)        Python, single numbers (and pandas Series)
#        np.round(arr, n)   NumPy arrays
#        Both round exact halves to the nearest EVEN digit (2.5 -> 2).
#
# 5. Edge cases worth naming in an interview:
#      - count = 0          -> []
#      - count = 1          -> [start]
#      - start > stop       -> counts down, e.g. 10 -> 0
#      - start == stop      -> the same value count times
#      - negative / float bounds -> work as normal
#
# 6. Related: NumPy Fundamentals - creating arrays (arange, linspace),
#    data_range, rounding with np.round.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    tests = [
        # (start, stop, count, expected, description)
        (0, 10, 5, [0.0, 2.5, 5.0, 7.5, 10.0], "given example"),
        (0, 10, 0, [], "count 0 -> []"),
        (3, 9, 1, [3.0], "count 1 -> [start]"),
        (10, 0, 5, [10.0, 7.5, 5.0, 2.5, 0.0], "start > stop (counts down)"),
        (4, 4, 3, [4.0, 4.0, 4.0], "start == stop"),
        (0, 1, 4, [0.0, 0.333333, 0.666667, 1.0], "rounded to 6 decimals"),
        (-5, 5, 3, [-5.0, 0.0, 5.0], "negative bounds"),
        (0.5, 1.5, 3, [0.5, 1.0, 1.5], "float bounds"),
        (0, 99, 100, [float(i) for i in range(100)], "max count 100"),
    ]

    passed = 0
    for start, stop, count, expected, desc in tests:
        got = evenly_spaced(start, stop, count)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 5 else f"[{len(got)} values]"
        print(f"{status}: {desc:<28} ({start}, {stop}, {count}) -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = evenly_spaced(0, 10, 5)
    assert type(out) is list, "plain list, not ndarray"
    assert all(type(v) is float for v in out), "plain Python floats"
    assert evenly_spaced(0, 10, 0) == [] and evenly_spaced(0, 10, 0) is not None
    print("PASS: list type / float items / [] not None")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()