"""
    Inner-Joining Two Tables on a Shared Key
    Medium
    GroupBy, Aggregation, Merge and Concat
    Write inner_merge_tables(records_a, records_b, on). Build two DataFrames
    and join them on the shared column on using pd.merge(..., how="inner"),
    keeping only rows whose key appears in BOTH tables. Return the joined
    table as a list of row dicts.

    Examples

    Arguments: [{'id': 1, 'name': 'Ravi'}, {'id': 2, 'name': 'Sam'}],
               [{'id': 1, 'amount': 500}], 'id'
    Returns: [{'id': 1, 'name': 'Ravi', 'amount': 500}]
    Explanation: Only id 1 exists in both tables, so only Ravi's row survives
    the inner join, now carrying the amount too.
    Constraints

    records_a and records_b are non-empty lists of dicts. Both contain the
    column named by on. 1 <= len(records_a), len(records_b) <= 500
"""
import pandas as pd


def inner_merge_tables(records_a, records_b, on):
    """
    inner merge on `on` -> list of row dicts.

    Time:  O(n + m) - hash join over n rows of A and m rows of B
           (more if a key repeats and rows multiply).
    Space: O(n + m) - for the DataFrames and the result.
    """
    # Step 0: both lists of dicts -> DataFrames.
    df1 = pd.DataFrame(records_a)
    df2 = pd.DataFrame(records_b)

    # Step 1: INNER join - keep only rows whose key is in BOTH tables.
    # df1.merge(df2, ...) is the same as pd.merge(df1, df2, ...).
    # No unmatched rows survive, so no NaN appears and int columns stay int
    # (500 stays 500, unlike the left join where it became 500.0).
    df3 = df1.merge(df2, on=on, how="inner")

    # Step 2: DataFrame -> list of row dicts ("records" is a keyword).
    return df3.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Inner vs left (compare with left_merge_with_missing):
#        inner -> only matching keys       -> Sam dropped, amount stays int
#        left  -> all rows of A            -> Sam kept with NaN, amount float
#    how="inner" is the DEFAULT, so df1.merge(df2, on=on) gives the same
#    result - writing it out makes the intent clear.
#
# 2. Two ways to write it (same result):
#        df1.merge(df2, on=on, how="inner")
#        pd.merge(df1, df2, on=on, how="inner")      <- form named in the task
#
# 3. Row order:
#      Inner join keeps the LEFT table's order of keys.
#
# 4. Duplicate keys:
#      If a key appears twice in B, the matching A row appears twice in the
#      result (one per match). 2 in A x 3 in B -> 6 rows for that key.
#
# 5. Same non-key column in both tables:
#      pandas adds suffixes name_x / name_y; change with suffixes=(...).
#
# 6. Keys with different column names:
#        pd.merge(df1, df2, left_on="cust_id", right_on="id")
#
# 7. Common mistakes:
#      - forgetting on=on               -> pandas joins on ALL shared columns
#      - how="left" by habit            -> unmatched rows kept with NaN
#      - to_dict() without "records"    -> {col: {index: value}}, wrong shape
#
# 8. Edge cases worth naming in an interview:
#      - no shared keys                 -> []
#      - every key matches              -> all rows, columns from both
#      - key repeated in B              -> left row duplicated
#      - key types differ (1 vs "1")    -> pandas raises ValueError
#        (can't merge int64 with str); convert one side with astype first


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    a = [{'id': 1, 'name': 'Ravi'}, {'id': 2, 'name': 'Sam'}]
    b = [{'id': 1, 'amount': 500}]
    big_a = [{'k': i, 'x': i} for i in range(500)]
    big_b = [{'k': i, 'y': i} for i in range(0, 500, 5)]

    tests = [
        # (records_a, records_b, on, expected, description)
        (a, b, 'id', [{'id': 1, 'name': 'Ravi', 'amount': 500}],
         "given example"),
        (a, [{'id': 9, 'amount': 1}], 'id', [], "no shared keys"),
        (a, [{'id': 2, 'amount': 7}, {'id': 1, 'amount': 5}], 'id',
         [{'id': 1, 'name': 'Ravi', 'amount': 5},
          {'id': 2, 'name': 'Sam', 'amount': 7}],
         "all match, left order kept"),
        ([{'id': 1}], [{'id': 1, 'v': 3}, {'id': 1, 'v': 4}], 'id',
         [{'id': 1, 'v': 3}, {'id': 1, 'v': 4}], "key repeated in B"),
        ([{'id': 1, 'n': 'a'}], [{'id': '1', 'v': 3}], 'id', None,
         "int vs str key"),
        (big_a, big_b, 'k', None, "max size n=500"),
    ]

    passed = 0
    for records_a, records_b, on, expected, desc in tests:
        try:
            got = inner_merge_tables(records_a, records_b, on)
        except ValueError:   # pandas refuses int vs str keys
            got = "ValueError"
        if desc == "int vs str key":
            ok = got == "ValueError" or got == []
        elif desc.startswith("max size"):
            ok = len(got) == 100
        else:
            ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if isinstance(got, str) or len(got) <= 2 \
            else f"[{len(got)} rows]"
        print(f"{status}: {desc:<30} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = inner_merge_tables(a, b, 'id')
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert type(out[0]['amount']) is int, "no NaN -> stays int"
    assert len(a) == 2 and 'amount' not in a[0], "input unchanged"
    print("PASS: return type / int kept / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()