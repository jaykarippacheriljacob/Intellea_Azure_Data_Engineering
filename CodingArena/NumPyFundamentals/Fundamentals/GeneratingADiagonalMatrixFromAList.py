"""
    Generating a Diagonal Matrix from a List
    Medium
    NumPy Fundamentals
    Write diagonal_matrix(values). Use np.diag(values) to build a square
    matrix with values along the main diagonal and zeros everywhere else, and
    return it as a list of lists.

    Examples

    Arguments: [1, 2, 3]
    Returns: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    Explanation: A 3x3 matrix with 1, 2, 3 down the diagonal and zeros
    elsewhere.
    Constraints

    values is a list of numbers. 0 <= len(values) <= 20
"""
import numpy as np


def diagonal_matrix(values):
    """
    Return an n x n matrix (list of lists) with `values` on the main diagonal
    and 0 everywhere else.

    Time:  O(n^2) - the matrix has n * n cells.
    Space: O(n^2)
    """
    # np.diag(1-D list/array) -> BUILDS a square 2-D matrix:
    #     [1, 2, 3] -> [[1, 0, 0],
    #                   [0, 2, 0],
    #                   [0, 0, 3]]
    # Main diagonal = positions [0][0], [1][1], [2][2] (row index == column).
    arr = np.diag(values)

    # .tolist() on a 2-D array -> list of lists, with plain Python numbers
    # (np.int64 -> int, np.float64 -> float).
    return arr.tolist()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no NumPy):
#        n = len(values)
#        return [[values[i] if i == j else 0 for j in range(n)]
#                for i in range(n)]
#
# 2. np.diag does TWO different jobs (interview trap):
#        np.diag([1, 2, 3])            1-D in -> BUILDS a 3x3 diagonal matrix
#        np.diag([[1, 2], [3, 4]])     2-D in -> EXTRACTS the diagonal [1, 4]
#
# 3. Other diagonal options:
#        np.diag(values, k=1)   values on the diagonal ABOVE the main one
#                               (matrix becomes (n+1) x (n+1))
#        np.diag(values, k=-1)  the diagonal BELOW
#        np.eye(3)              identity matrix (1s on the diagonal, floats)
#        np.identity(3)         same as np.eye(3)
#        np.trace(m)            sum of the diagonal
#
# 4. Types (tested):
#        [1, 2, 3]  -> int64   -> [[1, 0, 0], ...]       zeros are ints
#        [1.5, 2]   -> float64 -> [[1.5, 0.0], [0.0, 2.0]]  zeros are floats
#        []         -> shape (0, 0) -> []
#
# 5. Edge cases worth naming in an interview:
#      - empty list      -> [] (a 0 x 0 matrix)
#      - one value [5]   -> [[5]]
#      - zeros in values -> that diagonal cell is 0 too
#      - negative / float values -> work as normal
#
# 6. Related: LeetCode 1572 (Matrix Diagonal Sum), NumPy Fundamentals -
#    creating arrays (zeros, ones, eye) and 2-D indexing.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = list(range(1, 21))

    tests = [
        # (values, expected, description)
        ([1, 2, 3], [[1, 0, 0], [0, 2, 0], [0, 0, 3]], "given example"),
        ([], [], "empty list"),
        ([5], [[5]], "single value"),
        ([1.5, 2], [[1.5, 0.0], [0.0, 2.0]], "floats"),
        ([-1, 0, 4], [[-1, 0, 0], [0, 0, 0], [0, 0, 4]], "negative and zero"),
        ([7, 7], [[7, 0], [0, 7]], "repeated values"),
        (big, [[big[i] if i == j else 0 for j in range(20)] for i in range(20)],
         "max size 20 x 20"),
    ]

    passed = 0
    for values, expected, desc in tests:
        got = diagonal_matrix(values)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 3 else f"[{len(got)} x {len(got)} matrix]"
        print(f"{status}: {desc:<20} {values if len(values) <= 3 else '[1..20]'} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = diagonal_matrix([1, 2, 3])
    assert type(out) is list and all(type(r) is list for r in out), "list of lists"
    assert all(type(v) is int for r in out for v in r), "plain Python ints"
    assert len(out) == len(out[0]) == 3, "square"
    print("PASS: list of lists / plain ints / square")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()
    