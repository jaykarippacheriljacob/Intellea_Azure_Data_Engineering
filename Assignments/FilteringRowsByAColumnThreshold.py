"""
    Filtering Rows by a Column Threshold
    Easy
    Data Selection, Filtering and Data Cleaning
    Write filter_by_condition(records, column, threshold). Build a DataFrame
    and return only the rows where column is STRICTLY greater than threshold,
    as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 40}],
               'marks', 50
    Returns: [{'name': 'Ravi', 'marks': 88}]
    Explanation: Only Ravi's 88 clears a threshold of 50; Sam's 40 does not.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds
    numeric values. 1 <= len(records) <= 1000
"""
import pandas as pd


def filter_by_condition(records, column, threshold):
    """
    Return the rows of `records` where `column` > `threshold`, as a list of
    row dicts.

    Time:  O(n * k) - n rows, k columns (build, compare, convert back).
    Space: O(n * k) - for the DataFrame and the output list.
    """
    # Step 1: list of dicts -> DataFrame (each dict is one row).
    df = pd.DataFrame(records)

    # Step 2: boolean mask. df[column] > threshold gives True/False per row;
    # df[mask] keeps only the True rows. ">" is STRICT, so a value equal to
    # the threshold is dropped.
    filtered = df[df[column] > threshold]

    # REVISION 1 (bug fix): the original `df.to_dict()` returns a dict of
    # COLUMNS keyed by row index, e.g. {'name': {0: 'Ravi'}, 'marks': {0: 88}}.
    # The task wants a list of ROW dicts, so pass "records":
    # [{'name': 'Ravi', 'marks': 88}].
    return filtered.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas):
#        def filter_by_condition(records, column, threshold):
#            return [row for row in records if row[column] > threshold]
#
# 2. to_dict() and its `orient` argument (common trap):
#    to_dict() can lay out the same table in different shapes. The text you
#    pass in chooses the shape. For the table
#           name  marks
#        0  Ravi     88
#        1   Sam     40
#
#        df.to_dict()           -> {'name': {0: 'Ravi', 1: 'Sam'},
#                                   'marks': {0: 88, 1: 40}}
#                                  default "dict": by column, then row index
#        df.to_dict("list")     -> {'name': ['Ravi', 'Sam'], 'marks': [88, 40]}
#                                  by column
#        df.to_dict("records")  -> [{'name': 'Ravi', 'marks': 88},
#                                   {'name': 'Sam', 'marks': 40}]
#                                  by ROW: one dict per row  <- what we need
#        df.to_dict("index")    -> {0: {'name': 'Ravi', 'marks': 88}, 1: {...}}
#                                  by row index
#
#    IMPORTANT: "records" is a FIXED KEYWORD (a string option built into
#    pandas), NOT the function's variable `records`.
#        df.to_dict("records")   # correct: the text "records"
#        df.to_dict(records)     # wrong: passes your list -> error
#    In data terms a "record" = one row, so "records" means "one dict per
#    row". The argument is also called `records` because it is a list of
#    rows; the names match by meaning, not because pandas reads your
#    variable. Renaming the argument to `data` would still need
#    to_dict("records").
#
# 3. Other ways to filter in pandas:
#        df.loc[df[column] > threshold]
#        df.query(f"{column} > @threshold")   # column name must be valid
#
# 4. Edge cases worth naming in an interview:
#      - value equal to threshold  -> excluded (strictly greater)
#      - no row passes             -> [] (empty list)
#      - every row passes          -> all records back
#      - negative / float values   -> work as normal
#      - NaN in the column         -> NaN > x is False, so the row is dropped
#      - index after filtering is NOT reset (0, 2, 5...); to_dict("records")
#        ignores the index, so it doesn't matter here. Use
#        .reset_index(drop=True) if you keep working with the DataFrame.
#
# 5. Related problems: LeetCode 2883 (Drop Missing Data), LeetCode 1757
#    (Recyclable and Low Fat Products) - filtering with a boolean mask.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    students = [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 40}]
    big = [{'id': i, 'v': i} for i in range(1000)]

    tests = [
        # (records, column, threshold, expected, description)
        (students, 'marks', 50, [{'name': 'Ravi', 'marks': 88}],
         "given example"),
        (students, 'marks', 88, [],
         "equal is excluded"),
        (students, 'marks', 100, [],
         "no row passes"),
        (students, 'marks', 0, students,
         "all rows pass"),
        ([{'x': -5}, {'x': -1}, {'x': 3}], 'x', -2, [{'x': -1}, {'x': 3}],
         "negative threshold"),
        ([{'p': 1.5}, {'p': 2.25}], 'p', 2.0, [{'p': 2.25}],
         "floats"),
        ([{'a': 1, 'b': 9}, {'a': 7, 'b': 2}], 'a', 5, [{'a': 7, 'b': 2}],
         "other columns kept"),
        (big, 'v', 997, [{'id': 998, 'v': 998}, {'id': 999, 'v': 999}],
         "max size n=1000"),
    ]

    passed = 0
    for records, column, threshold, expected, desc in tests:
        got = filter_by_condition(records, column, threshold)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<20} {column} > {threshold} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = filter_by_condition(students, 'marks', 50)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert type(out[0]['marks']) is int, "plain Python int, not numpy"
    assert students[1] == {'name': 'Sam', 'marks': 40}, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()