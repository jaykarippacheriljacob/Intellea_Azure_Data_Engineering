"""
    Binning Values into Categories by Threshold
    Medium
    NumPy for Data Analytics
    Write bin_values(values, boundaries, labels). Use
    np.searchsorted(boundaries, values, side="right") to find which bin each
    value falls into, and return the corresponding entry from labels for
    each value, as a plain list. A value exactly equal to a boundary falls
    into the NEXT (higher) bin.

    Examples

    Arguments: [30, 60, 90], [50, 80], ['Low', 'Mid', 'High']
    Returns: ['Low', 'Mid', 'High']
    Explanation: 30 is Low, 60 is Mid, 90 is High.
    Constraints

    values is a list of numbers. boundaries is a sorted list of numbers with
    len(labels) == len(boundaries) + 1. 0 <= len(values) <= 1000
"""
import numpy as np


def bin_values(values, boundaries, labels):
    """
    Return the label of the bin each value falls into, as a plain list.
    A value equal to a boundary goes into the higher bin.

    Time:  O(n log b) - binary search per value (b = number of boundaries).
    Space: O(n)
    """
    # Step 1: lists -> arrays.
    arr = np.asarray(values)
    b = np.asarray(boundaries)
    l = np.asarray(labels)

    # Step 2: BIN NUMBERS. For each value: how many boundaries does it pass?
    # Boundaries [50, 80] cut the line into 3 bins:
    #        Low  |  Mid  |  High
    #     -------50------80-------
    #     bin 0   bin 1    bin 2
    # side="right": a value EQUAL to a boundary counts as having passed it
    #   -> 50 is bin 1 (Mid), 80 is bin 2 (High) - "falls into the NEXT bin".
    # side="left" would keep it in the lower bin (50 -> Low).
    idx = np.searchsorted(b, arr, side="right")

    # Step 3: bin numbers -> labels, via fancy indexing (l[[0, 1, 2]]).
    # FIRST ATTEMPT: returned the bin numbers ([0, 1, 2]) and never used
    # labels - step 3 was missing.
    #
    # Step 4: .tolist() -> plain list of Python strings.
    return l[idx].tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy) - bisect is Python's binary search:
#        from bisect import bisect_right
#        return [labels[bisect_right(boundaries, v)] for v in values]
#    bisect_right == searchsorted(side="right").
#
# 2. Same job, other tools:
#        np.digitize(arr, b)             -> same bin numbers as side="right"
#        pd.cut(s, bins=[-inf, 50, 80, inf], labels=[...], right=False)
#                                        -> pandas, returns a Categorical
#        np.select([v < 50, v < 80], ["Low", "Mid"], default="High")
#
# 3. Why len(labels) == len(boundaries) + 1: n boundaries cut the number
#    line into n + 1 bins. With fewer labels, the top bin number would be
#    out of range -> IndexError.
#
# 4. searchsorted needs SORTED boundaries - it uses binary search, so an
#    unsorted list gives wrong bins without any error.
#
# 5. Edge cases worth naming in an interview (tested):
#      - value == boundary   -> NEXT bin (50 -> Mid, 80 -> High)
#      - below all bounds    -> first label
#      - above all bounds    -> last label
#      - empty values        -> []
#      - one boundary        -> two bins
#      - floats / negatives  -> work as normal
#
# 6. Related: bucket_by_condition (two bins with np.where); this is the
#    general version for any number of bins.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    lmh = ['Low', 'Mid', 'High']
    big = list(range(1000))

    tests = [
        # (values, boundaries, labels, expected, description)
        ([30, 60, 90], [50, 80], lmh, ['Low', 'Mid', 'High'], "given example"),
        ([50, 80], [50, 80], lmh, ['Mid', 'High'], "on a boundary -> next bin"),
        ([49.99, 79.99], [50, 80], lmh, ['Low', 'Mid'], "just below boundaries"),
        ([-100, 1000], [50, 80], lmh, ['Low', 'High'], "far below / far above"),
        ([], [50, 80], lmh, [], "empty values"),
        ([5, 10, 15], [10], ['Small', 'Big'], ['Small', 'Big', 'Big'],
         "one boundary"),
        ([-5, 0, 5], [0], ['Neg', 'NonNeg'], ['Neg', 'NonNeg', 'NonNeg'],
         "zero boundary"),
        ([90, 30, 60, 30], [50, 80], lmh, ['High', 'Low', 'Mid', 'Low'],
         "order kept, repeats"),
        (big, [500], ['A', 'B'], ['A'] * 500 + ['B'] * 500, "max size n=1000"),
    ]

    passed = 0
    for values, bounds, labels, expected, desc in tests:
        got = bin_values(values, bounds, labels)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 4 else f"[{len(got)} labels]"
        print(f"{status}: {desc:<28} {values if len(values) <= 4 else '[0..999]'} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = bin_values([30, 60, 90], [50, 80], lmh)
    assert type(out) is list and all(type(v) is str for v in out), "plain str"
    print("PASS: plain list of str")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()