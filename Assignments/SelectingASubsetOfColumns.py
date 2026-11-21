"""
    Selecting a Subset of Columns
    Easy
    Introduction to Pandas and Data Exploration
    Write select_columns(records, columns). Build a DataFrame from records and
    return only the columns listed in columns, in that order, as a list of row
    dicts (df[columns].to_dict('records')).

    Examples

    Arguments: [{'name': 'Ravi', 'age': 23, 'marks': 88},
                {'name': 'Sam', 'age': 25, 'marks': 91}], ['name', 'marks']
    Returns: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 91}]
    Explanation: Only the name and marks columns are kept, age is dropped
    entirely.
    Constraints

    records is a non-empty list of dicts sharing the same keys. Every name in
    columns is an actual key in records. 1 <= len(records) <= 500
"""
import pandas as pd


def select_columns(records, columns):
    """
    Return the rows of `records` with only the keys in `columns`, in the
    order given by `columns`, as a list of dicts.

    Time:  O(n * k) - n rows, k = number of columns (building the DataFrame
           and converting back both touch every cell).
    Space: O(n * k) - for the DataFrame and the output list.
    """
    # Step 1: list of dicts -> DataFrame. Each dict becomes one row and each
    # key becomes a column name.
    df = pd.DataFrame(records)

    # Step 2: df[columns] - `columns` is a LIST, so this returns a DataFrame
    # (not a Series) with the columns in exactly the order listed.
    # A single name like df['name'] would return a Series instead.
    #
    # Step 3: .to_dict('records') -> DataFrame back to a list of row dicts,
    # which is the format the task wants returned.
    return df[columns].to_dict('records')


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas) - same result, no DataFrame needed:
#        def select_columns(records, columns):
#            return [{c: row[c] for c in columns} for row in records]
#    A dict comprehension keeps insertion order (Python 3.7+), so the
#    column order matches `columns` too.
#
# 2. Other pandas ways to pick columns:
#        df.loc[:, columns]          # label-based, same result
#        df.filter(items=columns)    # also keeps the given order
#        df.drop(columns=['age'])    # the opposite: say what to REMOVE
#
# 3. Single vs double brackets (common beginner trap):
#        df['name']      -> Series   (one column)
#        df[['name']]    -> DataFrame (one column, still a table)
#
# 4. Edge cases worth naming in an interview:
#      - columns in a different order than the data -> output follows
#        `columns` (handled)
#      - all columns selected -> same data back
#      - single row / single column -> still a list of dicts
#      - empty `columns` list -> pandas returns [] (NOT [{}, {}] - a
#        DataFrame with no columns gives no records). The pure-Python
#        version above returns [{}, {}] instead, so the two differ here.
#      - a name not in records -> KeyError (ruled out by the constraints)
#      - records with different keys -> pandas fills gaps with NaN (also
#        ruled out by the constraints)
#
# 5. Related problems: LeetCode 2880 (Select Data), LeetCode 2878 (Get the
#    Size of a DataFrame), LeetCode 2877 (Create a DataFrame from List) -
#    the "Introduction to Pandas" study plan.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    students = [
        {'name': 'Ravi', 'age': 23, 'marks': 88},
        {'name': 'Sam', 'age': 25, 'marks': 91},
    ]
    big = [{'id': i, 'x': i * 2, 'y': i * 3} for i in range(500)]

    tests = [
        # (records, columns, expected, description)
        (students, ['name', 'marks'],
         [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 91}],
         "given example"),
        (students, ['marks', 'name'],
         [{'marks': 88, 'name': 'Ravi'}, {'marks': 91, 'name': 'Sam'}],
         "reordered columns"),
        (students, ['age'],
         [{'age': 23}, {'age': 25}],
         "single column"),
        (students, ['name', 'age', 'marks'],
         students,
         "all columns"),
        ([{'a': 1, 'b': 2}], ['b'],
         [{'b': 2}],
         "single row"),
        ([{'a': 1.5, 'b': 'x', 'c': True}], ['c', 'a'],
         [{'c': True, 'a': 1.5}],
         "mixed value types"),
        (students, [],
         [],
         "empty columns list"),
        (big, ['y', 'id'],
         [{'y': i * 3, 'id': i} for i in range(500)],
         "max size n=500"),
    ]

    passed = 0
    for records, columns, expected, desc in tests:
        got = select_columns(records, columns)
        # == on dicts ignores key order, so also compare the key order.
        ok = got == expected and all(
            list(g.keys()) == list(e.keys()) for g, e in zip(got, expected)
        )
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<20} {columns} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = select_columns(students, ['name', 'marks'])
    assert type(out) is list and type(out[0]) is dict, "list of dicts"
    assert type(out[0]['marks']) is int, "plain Python int, not numpy"
    assert 'age' not in out[0], "unselected column must be dropped"
    assert students[0] == {'name': 'Ravi', 'age': 23, 'marks': 88}, \
        "input must not be modified"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()