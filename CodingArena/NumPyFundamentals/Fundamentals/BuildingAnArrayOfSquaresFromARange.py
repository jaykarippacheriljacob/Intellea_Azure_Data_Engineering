"""
    Building an Array of Squares from a Range
    Easy
    NumPy Fundamentals
    Write squares_up_to(n). Build the array 0, 1, ..., n-1 with np.arange(n),
    square every element, and return the result as a plain list.

    Examples

    Arguments: 5
    Returns: [0, 1, 4, 9, 16]
    Explanation: 0 through 4, squared: [0, 1, 4, 9, 16].
    Constraints

    0 <= n <= 1000
"""
import numpy as np


def squares_up_to(n):
    """
    Return [0**2, 1**2, ..., (n-1)**2] as a plain Python list.

    Time:  O(n)
    Space: O(n)
    """
    # Step 1: np.arange(n) -> [0, 1, ..., n-1]  (n itself is EXCLUDED,
    # like Python's range). np.arange(0) -> empty array.
    arr = np.arange(n)

    # Step 2: vectorised maths - ** 2 squares EVERY element at once, no loop.
    arr = arr ** 2

    # Step 3: array -> plain list of Python ints.
    return arr.tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. One-liner:
#        return (np.arange(n) ** 2).tolist()
#    Brackets matter: np.arange(n) ** 2 .tolist() would call tolist on 2.
#
# 2. Pure-Python version (no NumPy):
#        return [i * i for i in range(n)]
#
# 3. Other ways to square an array (same result):
#        arr ** 2      np.square(arr)      arr * arr      np.power(arr, 2)
#
# 4. Vectorised vs loop: arr ** 2 runs in C over the whole array; a Python
#    for-loop squares one number at a time and is much slower for big n.
#
# 5. arange reminders:
#        np.arange(5)          -> [0 1 2 3 4]        stop excluded
#        np.arange(1, 6)       -> [1 2 3 4 5]        start, stop
#        np.arange(0, 10, 2)   -> [0 2 4 6 8]        start, stop, step
#        np.linspace(0, 10, 5) -> COUNT, stop included (see evenly_spaced)
#
# 6. Overflow note: int64 holds up to ~9.2e18, so squares only overflow
#    past ~3 billion. With n <= 1000 the largest value is 999**2 = 998001.
#
# 7. Edge cases worth naming in an interview:
#      - n = 0  -> []
#      - n = 1  -> [0]
#      - result is ints, not floats (arange of ints stays int)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    tests = [
        # (n, expected, description)
        (5, [0, 1, 4, 9, 16], "given example"),
        (0, [], "n = 0 -> []"),
        (1, [0], "n = 1 -> [0]"),
        (2, [0, 1], "n = 2"),
        (10, [i * i for i in range(10)], "n = 10"),
        (1000, [i * i for i in range(1000)], "max n = 1000"),
    ]

    passed = 0
    for n, expected, desc in tests:
        got = squares_up_to(n)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 5 else f"[{len(got)} values, last {got[-1]}]"
        print(f"{status}: {desc:<16} n={n} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = squares_up_to(5)
    assert type(out) is list, "plain list"
    assert all(type(v) is int for v in out), "plain Python ints, not np.int64"
    print("PASS: list type / plain ints")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()
    