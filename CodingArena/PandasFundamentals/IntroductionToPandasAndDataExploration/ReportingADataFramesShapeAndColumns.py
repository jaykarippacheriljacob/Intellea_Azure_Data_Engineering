"""
    Reporting a DataFrame's Shape and Columns
    Easy
    Introduction to Pandas and Data Exploration
    Write describe_dataframe(records). records is a list of row dicts (all
    sharing the same keys). Build a DataFrame with pd.DataFrame(records) and
    return {"shape": [rows, cols], "columns": [...]} describing it.

    Examples

    Arguments: [{'name': 'Ravi', 'age': 23, 'marks': 88},
                {'name': 'Sam', 'age': 25, 'marks': 91}]
    Returns: {'shape': [2, 3], 'columns': ['name', 'age', 'marks']}
    Explanation: Two rows and three columns: shape [2, 3],
    columns ['name', 'age', 'marks'].
    Constraints

    records is a non-empty list of dicts that all share the same keys.
    1 <= len(records) <= 500
"""
import pandas as pd


def describe_dataframe(records):
    """
    Return {"shape": [rows, cols], "columns": [names...]} for the table
    built from `records`.

    Time:  O(n * k) - building the DataFrame touches every cell.
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame (each dict is one row).
    df = pd.DataFrame(records)

    # Step 2: df.shape is a TUPLE (rows, cols). Unpacking it into two names
    # works; the brackets around (row, col) are optional.
    rows, cols = df.shape

    # REVISION 1 (bug fix): df.columns is a pandas Index, not a list:
    #     Index(['name', 'age', 'marks'], dtype='str')
    # Index == list compares element by element and returns an ARRAY of
    # True/False ([True True True]). Inside a dict comparison that raises
    # "ValueError: The truth value of an array ... is ambiguous", so the
    # checker crashes. .tolist() turns it into a plain Python list:
    # ['name', 'age', 'marks'].
    columns = df.columns.tolist()

    # The task wants shape as a LIST [rows, cols], not the tuple (rows, cols).
    return {"shape": [rows, cols], "columns": columns}


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. One-line version:
#        df = pd.DataFrame(records)
#        return {"shape": list(df.shape), "columns": list(df.columns)}
#    list(df.shape) turns the tuple (2, 3) into [2, 3];
#    list(df.columns) works like .tolist().
#
# 2. Pure-Python version (no pandas):
#        return {"shape": [len(records), len(records[0])],
#                "columns": list(records[0].keys())}
#    Works because every dict shares the same keys (constraint).
#
# 3. Types to remember:
#        df.shape            -> tuple  (2, 3)
#        df.columns          -> Index  Index(['name', 'age', 'marks'])
#        df.columns.tolist() -> list   ['name', 'age', 'marks']
#        df.shape[0]/len(df) -> rows,  df.shape[1] -> columns
#
# 4. Rows/cols from df.shape are plain Python ints, so they compare and
#    print normally.
#
# 5. Edge cases worth naming in an interview:
#      - single row        -> [1, k]
#      - single column     -> [n, 1]
#      - column order      -> same as the keys of the first dict
#      - dicts with different keys (ruled out here) -> pandas adds the
#        missing keys as columns filled with NaN
#
# 6. Related problems: LeetCode 2878 (Get the Size of a DataFrame),
#    LeetCode 2877 (Create a DataFrame from List).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    students = [{'name': 'Ravi', 'age': 23, 'marks': 88},
                {'name': 'Sam', 'age': 25, 'marks': 91}]
    big = [{'id': i, 'x': i * 2} for i in range(500)]

    tests = [
        # (records, expected, description)
        (students, {'shape': [2, 3], 'columns': ['name', 'age', 'marks']},
         "given example"),
        ([{'a': 1}], {'shape': [1, 1], 'columns': ['a']},
         "single row, single column"),
        ([{'x': 1}, {'x': 2}, {'x': 3}], {'shape': [3, 1], 'columns': ['x']},
         "single column"),
        ([{'b': 1, 'a': 2, 'c': 3}], {'shape': [1, 3], 'columns': ['b', 'a', 'c']},
         "key order kept"),
        (big, {'shape': [500, 2], 'columns': ['id', 'x']},
         "max size n=500"),
    ]

    passed = 0
    for records, expected, desc in tests:
        got = describe_dataframe(records)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<26} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = describe_dataframe(students)
    assert type(out["shape"]) is list, "shape must be a list, not a tuple"
    assert type(out["columns"]) is list, "columns must be a list, not an Index"
    assert all(type(v) is int for v in out["shape"]), "plain Python ints"
    print("PASS: return types (list, list, int)")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()