"""
    Extracting Every Other Element
    Easy
    Array Indexing, Slicing & Operations
    Write every_other(values). Return every second element of values, starting from the first, using numpy slicing (arr[::2]).

    Examples

    Arguments: [1, 2, 3, 4, 5]
    Returns: [1, 3, 5]
    Explanation: Every second element starting at index 0: [1, 3, 5].
    Constraints

    values is a list of numbers. 0 <= len(values) <= 1000
"""
import numpy as np


def every_other(values):
    """
    Return every second element of `values`, starting at index 0.

    Slice syntax refresher:  arr[start:stop:step]
        start -> omitted, so it defaults to 0 (the first element)
        stop  -> omitted, so it defaults to the end of the array
        step  -> 2, so we take index 0, 2, 4, ...

    Time:  O(n)  - the result has ceil(n / 2) elements to copy out.
    Space: O(n)  - for the numpy array and the returned list.
    """
    # REVISION 1: your original `values[::2]` passed because Python lists
    # also support [::2], but it never actually used numpy. Converting with
    # np.asarray makes the numpy slicing the task asked for real, and it also
    # accepts tuples or numpy arrays as input.
    arr = np.asarray(values)

    # The actual work: a numpy slice. On a numpy array this returns a *view*
    # (no copy yet) - numpy just changes the stride to "skip one element".
    result = arr[::2]

    # REVISION 2: return a plain Python list so the output matches the
    # expected format exactly ([1, 3, 5], not array([1, 3, 5])).
    # .tolist() also turns numpy scalars (np.int64) back into normal ints.
    return result.tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no numpy) - simplest and fastest for small inputs:
#        return values[::2]
#    For n <= 1000, converting to numpy costs more than the slice saves.
#    numpy only pays off for large arrays or when you keep working in numpy.
#
# 2. Interview "no slicing allowed" version - two-pointer / index-step loop:
#        return [values[i] for i in range(0, len(values), 2)]
#
# 3. Follow-up: "every k-th element starting at index s"
#        def every_kth(values, k, start=0):
#            return np.asarray(values)[start::k].tolist()
#    Edge case: k must be != 0 (step 0 raises ValueError); a negative k
#    walks backwards, e.g. [::-2] gives every other element from the end.
#
# 4. Follow-up: "the OTHER half" (odd indices) -> values[1::2]
#    Splitting into even/odd index halves is a common sub-step in problems
#    like LeetCode 2149 (Rearrange Array Elements by Sign) or merging/
#    interleaving arrays (LeetCode 1470 Shuffle the Array, reversed).
#
# 5. Memory note: arr[::2] on a numpy array is a VIEW. If you modify it,
#    the original array changes too. Use arr[::2].copy() if you need an
#    independent array and plan to mutate it.
#
# 6. Empty input: np.asarray([]) gives a float array of length 0; slicing
#    it returns [], so no special-case `if not values` check is needed.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    tests = [
        # (input, expected, description)
        ([1, 2, 3, 4, 5], [1, 3, 5], "given example (odd length)"),
        ([1, 2, 3, 4], [1, 3], "even length"),
        ([], [], "empty list"),
        ([7], [7], "single element"),
        ([7, 8], [7], "two elements"),
        ([-1, -2, -3, -4, -5], [-1, -3, -5], "negative numbers"),
        ([0, 0, 0], [0, 0], "all zeros"),
        ([1.5, 2.5, 3.5], [1.5, 3.5], "floats"),
        ([5, 5, 5, 5, 5, 5], [5, 5, 5], "duplicates"),
        (list(range(1000)), list(range(0, 1000, 2)), "max size n=1000"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = every_other(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = values if len(values) <= 8 else f"[{len(values)} items]"
        print(f"{status}: {desc:<28} every_other({shown}) -> "
              f"{got if len(got) <= 8 else f'[{len(got)} items]'}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = every_other([1, 2, 3])
    assert isinstance(out, list), "should return a plain list"
    assert all(type(x) is int for x in out), "should contain plain ints"
    assert every_other((1, 2, 3)) == [1, 3], "tuples should work too"
    assert every_other(np.array([4, 5, 6])) == [4, 6], "numpy input works"
    print("PASS: return type / alternate input types")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()