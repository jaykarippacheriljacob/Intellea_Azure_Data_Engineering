"""
    Finding the Top N Rows by a Column
    Medium
    Introduction to Pandas and Data Exploration
    Write top_n_by_column(records, column, n). Sort the DataFrame by column
    in DESCENDING order and return the first n rows as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 95},
                {'name': 'Ana', 'marks': 70}], 'marks', 2
    Returns: [{'name': 'Sam', 'marks': 95}, {'name': 'Ravi', 'marks': 88}]
    Explanation: The two highest scorers, Sam (95) then Ravi (88), come back
    in descending order.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds
    numeric values. 0 <= n. 1 <= len(records) <= 500
"""
import pandas as pd


def top_n_by_column(records, column, n):
    """
    Return the n rows with the largest values in `column`, largest first,
    as a list of row dicts.

    Time:  O(r log r) - sorting r rows.
    Space: O(r * k)   - DataFrame of r rows, k columns.
    """
    # Step 1: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 2: sort by the column NAME (a string), largest first.
    #   df.sort_values(df[column])  -> wrong: passes the VALUES -> KeyError
    #   df.sort_values(column)      -> right: passes the NAME "marks"
    #
    # REVISION 1: kind="stable". The default sort (quicksort) does not keep
    # the original order of rows with EQUAL values. Tested on 20 rows with
    # ties: default gave B, D, H, F, R... ; stable gave B, D, F, H, J...
    # (original order). Stable makes the result predictable.
    ordered = df.sort_values(column, ascending=False, kind="stable")

    # Step 3: first n rows. head(0) -> empty; head(n > rows) -> all rows,
    # no error.
    top = ordered.head(n)

    # Step 4: DataFrame -> list of row dicts ("records" is a keyword).
    return top.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. pandas shortcut:
#        df.nlargest(n, column).to_dict("records")
#    nlargest keeps the first occurrence on ties (keep="first") and can be
#    faster than a full sort when n is small. nsmallest is the opposite.
#
# 2. Pure-Python version (no pandas):
#        return sorted(records, key=lambda r: r[column], reverse=True)[:n]
#    Python's sorted() is always stable, so ties keep their order.
#
# 3. sort_values reminders:
#        sort_values("col")                      ascending (default)
#        sort_values("col", ascending=False)     descending
#        sort_values(["a", "b"], ascending=[False, True])  tie-break by b
#        kind="stable"                           equal values keep order
#        na_position="first"/"last"              where NaN goes (last default)
#
# 4. Index after sorting is shuffled (e.g. 1, 0, 2). to_dict("records")
#    ignores it; use .reset_index(drop=True) if you keep working with df.
#
# 5. Edge cases worth naming in an interview:
#      - n = 0            -> [] (constraint allows it)
#      - n > rows         -> all rows, sorted
#      - ties             -> original order with kind="stable"
#      - negative values  -> sorted normally
#      - floats           -> sorted normally
#
# 6. Related problems: LeetCode 176 (Second Highest Salary),
#    LeetCode 2891 (Method Chaining) - sort then take the top rows.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    students = [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 95},
                {'name': 'Ana', 'marks': 70}]
    ties = [{'id': i, 'v': 9 if i % 2 else 5} for i in range(20)]
    big = [{'id': i, 'v': i % 97} for i in range(500)]

    tests = [
        # (records, column, n, expected, description)
        (students, 'marks', 2,
         [{'name': 'Sam', 'marks': 95}, {'name': 'Ravi', 'marks': 88}],
         "given example"),
        (students, 'marks', 0, [], "n = 0"),
        (students, 'marks', 1, [{'name': 'Sam', 'marks': 95}], "n = 1"),
        (students, 'marks', 10,
         [{'name': 'Sam', 'marks': 95}, {'name': 'Ravi', 'marks': 88},
          {'name': 'Ana', 'marks': 70}],
         "n > rows"),
        ([{'x': -5}, {'x': -1}, {'x': -9}], 'x', 2, [{'x': -1}, {'x': -5}],
         "negative values"),
        ([{'p': 1.5}, {'p': 2.25}, {'p': 0.5}], 'p', 2, [{'p': 2.25}, {'p': 1.5}],
         "floats"),
        (ties, 'v', 4, [{'id': 1, 'v': 9}, {'id': 3, 'v': 9},
                        {'id': 5, 'v': 9}, {'id': 7, 'v': 9}],
         "ties keep input order"),
        (big, 'v', 3, sorted(big, key=lambda r: r['v'], reverse=True)[:3],
         "max size n=500"),
    ]

    passed = 0
    for records, column, n, expected, desc in tests:
        got = top_n_by_column(records, column, n)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 3 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<22} n={n} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = top_n_by_column(students, 'marks', 2)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert students[0] == {'name': 'Ravi', 'marks': 88}, "input unchanged"
    assert out == pd.DataFrame(students).nlargest(2, 'marks').to_dict("records")
    print("PASS: return type / input untouched / matches nlargest")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()