"""
    Rounding Every Value to a Fixed Number of Decimals
    Easy
    Data Transformation & Statistical Analysis
    Write round_values(values, decimals). Use np.round(arr, decimals) to
    round every value in values to decimals decimal places, and return the
    result as a plain list.

    Examples

    Arguments: [3.14159, 2.71828], 2
    Returns: [3.14, 2.72]
    Explanation: Rounded to 2 decimals: [3.14, 2.72].
    Constraints

    values is a list of numbers. 0 <= decimals <= 10.
    0 <= len(values) <= 1000
"""
import numpy as np


def round_values(values, decimals):
    """
    Return every value rounded to `decimals` places, as a plain list.

    Time:  O(n)
    Space: O(n) - np.round returns a new array.
    """
    # Step 1: list -> array.
    arr = np.asarray(values)

    # Step 2: np.round(arr, decimals) rounds EVERY element at once
    # (vectorised) and returns a NEW array; the input is not changed.
    #
    # REVISION 1: the first version looped and wrote back into arr:
    #     for i in range(len(arr)):
    #         arr[i] = round(arr[i], decimals)
    # Same results (tested, incl. 2.675 and 0.125), but:
    #   - the task asks for np.round
    #   - the loop runs one value at a time in Python (slower)
    #   - writing into arr in place is the dtype trap from
    #     replace_if_negative: values are forced into arr's type
    #
    # Step 3: .tolist() -> plain Python list.
    return np.round(arr, decimals).tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        return [round(v, decimals) for v in values]
#
# 2. Same rounding, different spellings:
#        np.round(arr, 2)    arr.round(2)    np.around(arr, 2)
#        round(arr, 2)       -> TypeError on an ndarray (see evenly_spaced)
#
# 3. Rounding surprises (both Python and NumPy):
#      - exact halves go to the nearest EVEN digit ("banker's rounding"):
#            np.round([0.5, 1.5, 2.5]) -> [0.0, 2.0, 2.0]
#      - many decimals can't be stored exactly in binary:
#            2.675 is really 2.67499999...  -> round(2.675, 2) = 2.67 in
#            plain Python, while np.round gives 2.68 (it scales by 100
#            first). Tested here: 0.125 -> 0.12, 1.005 -> 1.0
#      - for money, use the decimal module instead of floats
#
# 4. Ints stay ints: np.round([1, 2], 2) -> [1, 2] (no .0 added).
#
# 5. Negative decimals round to tens, hundreds...:
#        np.round(1234, -2) -> 1200   (not needed here: 0 <= decimals)
#
# 6. Edge cases worth naming in an interview (tested):
#      - empty list       -> []
#      - decimals = 0     -> whole numbers as floats (2.5 -> 2.0)
#      - decimals > digits -> value unchanged
#      - negative values  -> rounded the same way


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [i / 7 for i in range(1000)]

    tests = [
        # (values, decimals, expected, description)
        ([3.14159, 2.71828], 2, [3.14, 2.72], "given example"),
        ([], 2, [], "empty list"),
        ([2.5, 3.5, 4.6], 0, [2.0, 4.0, 5.0], "decimals 0 (halves to even)"),
        ([1.23], 5, [1.23], "more decimals than digits"),
        ([-1.236, -0.004], 2, [-1.24, -0.0], "negative values"),
        ([1, 2], 2, [1, 2], "ints stay ints"),
        ([123.4567891234], 10, [123.4567891234], "decimals 10"),
        (big, 3, [round(v, 3) for v in big], "max size n=1000"),
    ]

    passed = 0
    for values, decimals, expected, desc in tests:
        got = round_values(values, decimals)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 3 else f"[{len(got)} values]"
        print(f"{status}: {desc:<28} d={decimals} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [3.14159, 2.71828]
    out = round_values(data, 2)
    assert type(out) is list and all(type(v) is float for v in out), "plain floats"
    assert data == [3.14159, 2.71828], "input unchanged"
    print("PASS: plain list of floats / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()